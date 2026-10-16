"""Noise-multiplier sweep for the private models (same protocol as run_all_experiments.py, 5 seeds)."""
import os, sys
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
import numpy as np, torch, pandas as pd
from data.download_benchmarks import load_statlog_heart, load_pima_diabetes, load_clinical_ehr_cohort
from src.train_eval import prepare_dataset_splits, train_model, evaluate_model_pipeline

MODELS = [
    ("Standard DP-SGD", dict(sc=False, swd=False, gcp=False)),
    ("DPSGD-F (simplified)", dict(sc=False, swd=False, gcp=False)),
    ("DP-EDL", dict(sc=False, swd=False, gcp=False)),
    ("TRACER-Med (Proposed)", dict(sc=True, swd=True, gcp=True)),
]
SIGMAS = [0.8, 1.0, 1.5, 2.0, 3.0]
KEYS = ["AUC", "ECE", "DP_Disparity", "EO_Disparity", "MIA_AUC", "Coverage_Disparity", "Avg_Set_Size", "Singlet_Rate", "Sensitivity", "Balanced_Acc", "AUC_Gap", "DP_Disparity_T05", "EO_Disparity_T05"]
rows = []
for loader in (load_statlog_heart, load_pima_diabetes, load_clinical_ehr_cohort):
    X, y, s, f, name = loader()
    epochs = 20 if X.shape[0] < 1000 else 12
    bs = 32 if X.shape[0] < 1000 else 128
    for sigma in SIGMAS:
        for m, c in MODELS:
            reps = []
            for seed in range(42, 47):
                sp = prepare_dataset_splits(X, y, s, seed=seed)
                torch.manual_seed(seed); np.random.seed(seed)
                model, eps, delta = train_model(model_type=m, splits=sp, epochs=epochs, lr=0.005, batch_size=bs,
                                                noise_multiplier=sigma, max_grad_norm=1.0, use_sc_dp=c["sc"],
                                                use_swd=c["swd"], lambda_fair=0.5)
                r, _ = evaluate_model_pipeline(model=model, model_type=m, splits=sp, eps=eps, delta=delta,
                                               conformal_alpha=0.10, conformal_eps=1.0, use_group_conformal=c["gcp"])
                reps.append(r)
            row = {"Dataset": name, "Sigma": sigma, "Model": m, "Epsilon": reps[0]["Epsilon"]}
            for k in KEYS:
                v = [r[k] for r in reps]; row[k + "_mean"], row[k + "_std"] = float(np.mean(v)), float(np.std(v))
            rows.append(row)
            print(name[:10], sigma, m, round(row["Epsilon"], 1), {k: round(row[k + "_mean"], 3) for k in KEYS[:5]}, flush=True)
            pd.DataFrame(rows).to_csv("experiments/results/noise_sweep.csv", index=False)
