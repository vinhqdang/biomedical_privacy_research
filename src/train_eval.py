"""
Comprehensive Training, Evaluation, and Benchmarking Pipeline
Trains TRACER-Med alongside competitive baselines:
1. Standard ERM (Vanilla DNN)
2. Standard DP-SGD (Abadi et al., 2016)
3. Fair-ERM (Sliced Wasserstein debiased DNN)
4. Standard Evidential Deep Learning (EDL)
5. DP-EDL (Vanilla DP applied to EDL)
6. TRACER-Med (Proposed unified trustworthy architecture)
7. Ablation variants (w/o SC-DP, w/o SWD, w/o DP-GCCP)
"""

import time
import math
import copy
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.func import functional_call
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

from src.edl_model import EvidentialNet, StandardClassifierNet, edl_total_loss
from src.fairness import sliced_wasserstein_distance, demographic_parity_loss
from src.dp_optimizer import VectorizedSCDPOptimizer, SubgroupCalibratedDPOptimizer
from src.conformal import DPGroupConditionalConformalPredictor
from src.xai import DualChannelDirichletExplainer, compute_explanation_infidelity, compute_faithfulness_correlation
from src.metrics import (
    compute_clinical_metrics, compute_calibration_metrics, 
    compute_fairness_metrics, compute_mia_risk
)

def prepare_dataset_splits(X, y, sensitive, test_size=0.25, cal_size=0.20, seed=42):
    """
    Creates stratified Train / Calibration / Test splits.
    Standardizes features using train set statistics only.
    """
    # 1. Split into Train+Cal vs Test
    X_train_cal, X_test, y_train_cal, y_test, s_train_cal, s_test = train_test_split(
        X, y, sensitive, test_size=test_size, random_state=seed, stratify=y
    )
    
    # 2. Split Train+Cal into Train and Conformal Calibration set
    rel_cal_size = cal_size / (1.0 - test_size)
    X_train, X_cal, y_train, y_cal, s_train, s_cal = train_test_split(
        X_train_cal, y_train_cal, s_train_cal, test_size=rel_cal_size, random_state=seed, stratify=y_train_cal
    )
    
    # Standardize
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_cal = scaler.transform(X_cal)
    X_test = scaler.transform(X_test)
    
    return {
        "X_train": torch.tensor(X_train, dtype=torch.float32),
        "y_train": torch.tensor(y_train, dtype=torch.long),
        "s_train": torch.tensor(s_train, dtype=torch.long),
        "X_cal": torch.tensor(X_cal, dtype=torch.float32),
        "y_cal": torch.tensor(y_cal, dtype=torch.long),
        "s_cal": torch.tensor(s_cal, dtype=torch.long),
        "X_test": torch.tensor(X_test, dtype=torch.float32),
        "y_test": torch.tensor(y_test, dtype=torch.long),
        "s_test": torch.tensor(s_test, dtype=torch.long),
        "scaler": scaler
    }

def _dp_mode(model_type, use_sc_dp):
    if "DPSGD-F" in model_type:
        return "dpsgdf"
    return "refdp" if use_sc_dp else "uniform"


def train_model(model_type, splits, epochs=25, lr=0.005, batch_size=64,
                noise_multiplier=0.8, max_grad_norm=1.0,
                use_sc_dp=True, use_swd=True, lambda_fair=0.5, pfld_lam_max=1.0, pfld_dual_step=0.5,
                device="cuda" if torch.cuda.is_available() else "cpu"):
    """
    Trains the specified model under the given privacy and fairness configuration.
    Private models use Poisson subsampling and the RDP accountant in src/dp_optimizer.py.
    """
    from torch.func import vmap, grad
    X_tr = splits["X_train"].to(device)
    y_tr = splits["y_train"].to(device)
    s_tr = splits["s_train"].to(device)

    in_features = X_tr.size(1)
    is_edl = "EDL" in model_type or "TRACER" in model_type
    is_dp = ("DP" in model_type or "TRACER" in model_type or "PF-LD" in model_type) and noise_multiplier > 0
    is_pfld = model_type.startswith("PF-LD")

    if is_edl:
        model = EvidentialNet(in_features=in_features, hidden_dim=64, num_classes=2).to(device)
    else:
        model = StandardClassifierNet(in_features=in_features, hidden_dim=64, num_classes=2).to(device)
    optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=1e-4)

    dp_opt = None
    if is_dp:
        dp_opt = VectorizedSCDPOptimizer(model=model, optimizer=optimizer, max_grad_norm=max_grad_norm,
                                         noise_multiplier=noise_multiplier, mode=_dp_mode(model_type, use_sc_dp))

    n_samples = X_tr.size(0)
    indices = np.arange(n_samples)
    q_rate = batch_size / n_samples

    # PF-LD (adapted from Tran et al., AAAI 2021): demographic-parity constraint on h(x) = P(Y=1|x)
    lam = torch.zeros(2, device=device)
    lam_max, dual_step, c_p = pfld_lam_max, pfld_dual_step, 1.0

    def h_fn(params, x_i):
        _, prob = functional_call(model, params, (x_i.unsqueeze(0),))
        return prob[0, 1]

    model.train()
    for epoch in range(epochs):
        np.random.shuffle(indices)
        for start_idx in range(0, n_samples, batch_size):
            if dp_opt is not None:
                batch_idx = np.nonzero(np.random.rand(n_samples) < q_rate)[0]
                if len(batch_idx) == 0:
                    continue
            else:
                batch_idx = indices[start_idx: start_idx + batch_size]
            bx, by, bs = X_tr[batch_idx], y_tr[batch_idx], s_tr[batch_idx]

            if dp_opt is not None:
                def loss_per_sample(params, x_i, y_i, s_i):
                    if is_edl:
                        ev, alpha, prob, unc = functional_call(model, params, (x_i.unsqueeze(0),))
                        total_l, _, _ = edl_total_loss(ev, alpha, y_i.unsqueeze(0), num_classes=2)
                    else:
                        logits, prob = functional_call(model, params, (x_i.unsqueeze(0),))
                        y_oh = (torch.arange(2, device=prob.device) == y_i).float()
                        total_l = -torch.sum(y_oh * torch.log(prob[0] + 1e-10))
                    if use_swd:
                        # PC-PAT: class-conditional prototype anchoring; the loss of record i depends on record i only
                        enc_params = {k.replace('encoder.', ''): v for k, v in params.items() if k.startswith('encoder.')}
                        h_i = functional_call(model.encoder, enc_params, (x_i.unsqueeze(0),))
                        proto = torch.sum(params['prototypes'] * (torch.arange(2, device=h_i.device) == y_i).float().unsqueeze(1), dim=0)
                        total_l = total_l + lambda_fair * 0.5 * torch.sum((h_i.squeeze(0) - proto) ** 2)
                    return total_l

                extra = None
                if is_pfld and len(batch_idx) > 2:
                    params_c = {k: v.detach() for k, v in model.named_parameters() if v.requires_grad}
                    gh = vmap(grad(h_fn), in_dims=(None, 0))(params_c, bx)
                    nrm = torch.sqrt(sum(torch.sum(g.reshape(len(batch_idx), -1) ** 2, dim=1) for g in gh.values()) + 1e-12)
                    fac = torch.clamp(c_p / nrm, max=1.0)
                    with torch.no_grad():
                        h_val = model(bx)[1][:, 1]
                    mu_p = h_val.mean()
                    cnt = torch.stack([(bs == g).sum() for g in (0, 1)])
                    delta_p = 2.0 * c_p * lam_max / max(int(cnt.min()) - 1, 1)
                    extra = {}
                    for name, g in gh.items():
                        gc = g * fac.view([-1] + [1] * (g.dim() - 1))
                        mean_p = gc.mean(0)
                        total = torch.zeros_like(mean_p)
                        for k in (0, 1):
                            if cnt[k] > 0:
                                mk = (bs == k)
                                sgn = torch.sign(mu_p - h_val[mk].mean())
                                total = total + lam[k] * sgn * (mean_p - gc[mk].mean(0))
                        extra[name] = total + torch.randn_like(total) * (noise_multiplier * delta_p)
                    dp_opt.add_constraint_release(q_rate, noise_multiplier)
                dp_opt.step(loss_per_sample, bx, by, bs, q_subsample=q_rate, expected_batch_size=batch_size,
                            extra_grads=extra)
            else:
                optimizer.zero_grad()
                if is_edl:
                    ev, alpha, prob, unc = model(bx)
                    loss, _, _ = edl_total_loss(ev, alpha, by, num_classes=2)
                else:
                    logits, prob = model(bx)
                    loss = nn.CrossEntropyLoss()(logits, by)
                if use_swd:
                    latent = model.forward_latent(bx)
                    from src.fairness import class_conditional_sliced_wasserstein_distance, equalized_odds_penalty
                    swd_loss = class_conditional_sliced_wasserstein_distance(latent, by, bs)
                    eo_loss = equalized_odds_penalty(prob, by, bs)
                    loss = loss + lambda_fair * (swd_loss + eo_loss)
                loss.backward()
                nn.utils.clip_grad_norm_(model.parameters(), max_grad_norm)
                optimizer.step()

        if dp_opt is not None and dp_opt.mode == "refdp" and epoch % 5 == 0:
            dp_opt.refresh_group_weights(X_tr, s_tr)
        if dp_opt is not None and is_pfld:
            with torch.no_grad():
                h_all = model(X_tr)[1][:, 1]
                n_g = torch.stack([(s_tr == g).sum() for g in (0, 1)])
                d_d = math.sqrt(2.0) / max(int(n_g.min()) - 1, 1)
                for k in (0, 1):
                    viol = torch.abs(h_all.mean() - h_all[s_tr == k].mean())
                    lam[k] = torch.clamp(lam[k] + dual_step * (viol + torch.randn(()).to(device) * noise_multiplier * d_d), 0.0, lam_max)
            dp_opt.add_constraint_release(1.0, noise_multiplier)

    eps, delta = (dp_opt.get_privacy_spent() if dp_opt is not None else (float('inf'), 0.0))
    return model, eps, delta

def evaluate_model_pipeline(model, model_type, splits, eps, delta, 
                            conformal_alpha=0.10, conformal_eps=1.0, 
                            use_group_conformal=True,
                            device="cuda" if torch.cuda.is_available() else "cpu"):
    """
    Comprehensive evaluation across all clinical trustworthiness dimensions.
    """
    model.eval()
    is_edl = "EDL" in model_type or "TRACER" in model_type
    
    def get_probs_and_unc(x_tensor):
        with torch.no_grad():
            x_dev = x_tensor.to(device)
            if is_edl:
                ev, alpha, prob, unc = model(x_dev)
                return prob.cpu().numpy(), unc.cpu().numpy()
            else:
                logits, prob = model(x_dev)
                unc = 1.0 - torch.max(prob, dim=1, keepdim=True)[0]
                return prob.cpu().numpy(), unc.cpu().numpy()
                
    train_probs, _ = get_probs_and_unc(splits["X_train"])
    cal_probs, _ = get_probs_and_unc(splits["X_cal"])
    test_probs, test_unc = get_probs_and_unc(splits["X_test"])
    
    y_test_np = splits["y_test"].numpy()
    s_test_np = splits["s_test"].numpy()
    
    # Operating point: Youden-J threshold chosen on the calibration split (no sensitive attribute,
    # same rule for every model), so decision-based metrics do not depend on each model's score scale.
    from sklearn.metrics import roc_curve
    fpr_c, tpr_c, thr_c = roc_curve(splits["y_cal"].numpy(), cal_probs[:, 1])
    j = tpr_c - fpr_c
    cand = np.where(j >= j.max() - 1e-12)[0]
    op_threshold = float(thr_c[cand[np.argmin(np.abs(thr_c[cand] - 0.5))]])
    op_threshold = min(max(op_threshold, 1e-6), 1.0 - 1e-6)

    # 1. Clinical utility metrics
    clinical_metrics = compute_clinical_metrics(y_test_np, test_probs[:, 1], threshold=op_threshold)
    
    # 2. Calibration metrics
    calib_metrics = compute_calibration_metrics(y_test_np, test_probs[:, 1])
    
    # 3. Fairness metrics
    fairness_metrics = compute_fairness_metrics(y_test_np, test_probs[:, 1], s_test_np, threshold=op_threshold)
    fm05 = compute_fairness_metrics(y_test_np, test_probs[:, 1], s_test_np, threshold=0.5)
    fairness_metrics["DP_Disparity_T05"] = fm05["DP_Disparity"]
    fairness_metrics["EO_Disparity_T05"] = fm05["EO_Disparity"]
    fairness_metrics["Op_Threshold"] = op_threshold
    
    # 4. Privacy metrics (Formal DP + Empirical MIA AUC)
    mia_risk = compute_mia_risk(train_probs[:, 1], splits["y_train"].numpy(),
                                test_probs[:, 1], y_test_np)
    privacy_metrics = {
        "Epsilon": float(eps),
        "Delta": float(delta),
        "MIA_AUC": float(mia_risk)
    }
    
    # 5. Conformal reliability (same private mechanism for every model that is trained privately)
    is_private_cp = is_dp_model = ("DP" in model_type or "TRACER" in model_type or "PF-LD" in model_type)
    def _run_cp(mode):
        cp = DPGroupConditionalConformalPredictor(alpha=conformal_alpha, epsilon_conf=conformal_eps, beta=0.05, mode=mode)
        cp.calibrate(cal_probs, splits["y_cal"].numpy(), splits["s_cal"].numpy(), private=is_private_cp,
                     use_group=use_group_conformal)
        sets, _ = cp.predict(test_probs, s_test_np, use_group_conditional=use_group_conformal)
        return cp.evaluate_coverage(sets, y_test_np, s_test_np)
    conformal_eval = _run_cp("plugin")
    conformal_pac = _run_cp("pac")

    # 6. XAI Evaluation (on subset of test samples for speed)
    eval_sub_n = min(30, splits["X_test"].size(0))
    x_sub = splits["X_test"][:eval_sub_n].to(device)
    
    if is_edl:
        explainer = DualChannelDirichletExplainer(model, steps=30)
        diag_attr, unc_attr = explainer.attribute(x_sub, target_class=1)
    else:
        # Standard input x gradient
        x_sub_grad = x_sub.clone().detach().requires_grad_(True)
        _, prob_sub = model(x_sub_grad)
        loss = prob_sub[:, 1].sum()
        loss.backward()
        diag_attr = (x_sub_grad * x_sub_grad.grad).detach().cpu().numpy()
        unc_attr = np.zeros_like(diag_attr)
        
    infidelity = compute_explanation_infidelity(model, x_sub, diag_attr, num_samples=15)
    faithfulness = compute_faithfulness_correlation(model, x_sub, diag_attr)
    
    xai_metrics = {
        "Infidelity": float(infidelity),
        "Faithfulness_Corr": float(faithfulness)
    }
    
    # Merge all results
    full_report = {
        "Model": model_type,
        **clinical_metrics,
        **calib_metrics,
        **fairness_metrics,
        **privacy_metrics,
        "Coverage_Marginal": conformal_eval["marginal_coverage"],
        "Coverage_Disparity": conformal_eval.get("max_coverage_disparity", 0.0),
        "Avg_Set_Size": conformal_eval["avg_set_size"],
        "Singlet_Rate": conformal_eval["singlet_rate"],
        "Coverage_Marginal_PAC": conformal_pac["marginal_coverage"],
        "Avg_Set_Size_PAC": conformal_pac["avg_set_size"],
        **xai_metrics
    }
    
    return full_report, model
