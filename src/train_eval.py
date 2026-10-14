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
import copy
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

from src.edl_model import EvidentialNet, StandardClassifierNet, edl_total_loss
from src.fairness import sliced_wasserstein_distance, demographic_parity_loss
from src.dp_optimizer import SubgroupCalibratedDPOptimizer
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

def train_model(model_type, splits, epochs=25, lr=0.005, batch_size=64,
                noise_multiplier=0.8, max_grad_norm=1.0, 
                use_sc_dp=True, use_swd=True, lambda_fair=0.5,
                device="cuda" if torch.cuda.is_available() else "cpu"):
    """
    Trains specified model architecture under given privacy and fairness constraints.
    """
    X_tr = splits["X_train"].to(device)
    y_tr = splits["y_train"].to(device)
    s_tr = splits["s_train"].to(device)
    
    in_features = X_tr.size(1)
    is_edl = "EDL" in model_type or "TRACER" in model_type
    is_dp = "DP" in model_type or "TRACER" in model_type
    
    if is_edl:
        model = EvidentialNet(in_features=in_features, hidden_dim=64, num_classes=2).to(device)
    else:
        model = StandardClassifierNet(in_features=in_features, hidden_dim=64, num_classes=2).to(device)
        
    optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=1e-4)
    
    if is_dp and noise_multiplier > 0:
        dp_opt = SubgroupCalibratedDPOptimizer(
            model=model,
            optimizer=optimizer,
            max_grad_norm=max_grad_norm,
            noise_multiplier=noise_multiplier,
            subgroup_balance=use_sc_dp,
            gamma=0.5
        )
    else:
        dp_opt = None
        
    n_samples = X_tr.size(0)
    indices = np.arange(n_samples)
    
    model.train()
    for epoch in range(epochs):
        np.random.shuffle(indices)
        
        for start_idx in range(0, n_samples, batch_size):
            batch_idx = indices[start_idx : start_idx + batch_size]
            bx = X_tr[batch_idx]
            by = y_tr[batch_idx]
            bs = s_tr[batch_idx]
            
            if dp_opt is not None:
                def loss_fn_single(x_i, y_i, s_i):
                    if is_edl:
                        ev, alpha, prob, unc = model(x_i)
                        total_l, _, _ = edl_total_loss(ev, alpha, y_i, num_classes=2)
                    else:
                        logits, prob = model(x_i)
                        total_l = nn.CrossEntropyLoss()(logits, y_i)
                    return total_l
                    
                q_sub = len(batch_idx) / n_samples
                dp_opt.step_with_microbatches(loss_fn_single, bx, by, bs, q_subsample=q_sub)
                
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
                    z0 = latent[bs == 0]
                    z1 = latent[bs == 1]
                    swd_loss = sliced_wasserstein_distance(z0, z1)
                    dp_loss = demographic_parity_loss(prob, bs)
                    loss = loss + lambda_fair * (swd_loss + dp_loss)
                    
                loss.backward()
                nn.utils.clip_grad_norm_(model.parameters(), max_grad_norm)
                optimizer.step()
                
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
    
    # 1. Clinical utility metrics
    clinical_metrics = compute_clinical_metrics(y_test_np, test_probs[:, 1])
    
    # 2. Calibration metrics
    calib_metrics = compute_calibration_metrics(y_test_np, test_probs[:, 1])
    
    # 3. Fairness metrics
    fairness_metrics = compute_fairness_metrics(y_test_np, test_probs[:, 1], s_test_np)
    
    # 4. Privacy metrics (Formal DP + Empirical MIA AUC)
    mia_risk = compute_mia_risk(train_probs[:, 1], splits["y_train"].numpy(),
                                test_probs[:, 1], y_test_np)
    privacy_metrics = {
        "Epsilon": float(eps),
        "Delta": float(delta),
        "MIA_AUC": float(mia_risk)
    }
    
    # 5. Conformal Reliability
    conformal = DPGroupConditionalConformalPredictor(
        alpha=conformal_alpha, epsilon_conf=conformal_eps, delta_conf=1e-5
    )
    is_private_cp = "DP" in model_type or "TRACER" in model_type
    conformal.calibrate(
        cal_probs, splits["y_cal"].numpy(), splits["s_cal"].numpy(), 
        private=is_private_cp
    )
    pred_sets, set_sizes = conformal.predict(
        test_probs, s_test_np, use_group_conditional=use_group_conformal
    )
    conformal_eval = conformal.evaluate_coverage(pred_sets, y_test_np, s_test_np)
    
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
        **xai_metrics
    }
    
    return full_report, model
