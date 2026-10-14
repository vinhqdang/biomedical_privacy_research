"""
Full Experimental Suite and Benchmark Orchestrator
Executes all models, baselines, and ablation studies across 3 biomedical benchmarks:
1. Statlog Heart Disease (Cardiovascular diagnosis)
2. Pima Indians Diabetes (Metabolic risk)
3. Multi-Center Clinical EHR Cohort (Acuity/Mortality)
Generates structured JSON/CSV results and publication-quality figures.
"""

import os
import json
import time
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import torch

from data.download_benchmarks import load_statlog_heart, load_pima_diabetes, load_clinical_ehr_cohort
from src.train_eval import prepare_dataset_splits, train_model, evaluate_model_pipeline
from src.xai import DualChannelDirichletExplainer

# Setup publication style
plt.style.use('seaborn-v0_8-whitegrid' if 'seaborn-v0_8-whitegrid' in plt.style.available else 'default')
plt.rcParams.update({
    'font.size': 11,
    'axes.labelsize': 12,
    'axes.titlesize': 13,
    'xtick.labelsize': 10,
    'ytick.labelsize': 10,
    'legend.fontsize': 10,
    'figure.titlesize': 14,
    'pdf.fonttype': 42,
    'ps.fonttype': 42
})

def run_dataset_benchmark(dataset_name, X, y, sensitive, feature_names, n_seeds=3):
    print(f"\n=======================================================")
    print(f"Executing Benchmark: {dataset_name}")
    print(f"Samples: {X.shape[0]}, Features: {X.shape[1]}, Sensitive Minority Rate: {min(sensitive.mean(), 1-sensitive.mean()):.2f}")
    print(f"=======================================================")
    
    models_to_test = [
        {"type": "Standard ERM", "is_dp": False, "noise": 0.0, "sc_dp": False, "swd": False, "group_cp": False},
        {"type": "Standard DP-SGD", "is_dp": True, "noise": 0.8, "sc_dp": False, "swd": False, "group_cp": False},
        {"type": "Fair-ERM", "is_dp": False, "noise": 0.0, "sc_dp": False, "swd": True, "group_cp": False},
        {"type": "Standard EDL", "is_dp": False, "noise": 0.0, "sc_dp": False, "swd": False, "group_cp": False},
        {"type": "DP-EDL", "is_dp": True, "noise": 0.8, "sc_dp": False, "swd": False, "group_cp": False},
        {"type": "TRACER-Med (Proposed)", "is_dp": True, "noise": 0.8, "sc_dp": True, "swd": True, "group_cp": True},
        {"type": "TRACER-Med w/o SC-DP", "is_dp": True, "noise": 0.8, "sc_dp": False, "swd": True, "group_cp": True},
        {"type": "TRACER-Med w/o SWD", "is_dp": True, "noise": 0.8, "sc_dp": True, "swd": False, "group_cp": True},
        {"type": "TRACER-Med w/o DP-GCCP", "is_dp": True, "noise": 0.8, "sc_dp": True, "swd": True, "group_cp": False},
    ]
    
    results = []
    trained_models = {}
    last_splits = None
    
    for cfg in models_to_test:
        m_type = cfg["type"]
        seed_metrics = []
        print(f"  --> Running {m_type} across {n_seeds} seeds...")
        
        for seed in range(42, 42 + n_seeds):
            splits = prepare_dataset_splits(X, y, sensitive, seed=seed)
            last_splits = splits
            
            # Select epochs based on dataset size for fast, robust convergence
            epochs = 20 if X.shape[0] < 1000 else 12
            batch_sz = 32 if X.shape[0] < 1000 else 128
            
            model, eps, delta = train_model(
                model_type=m_type,
                splits=splits,
                epochs=epochs,
                lr=0.005,
                batch_size=batch_sz,
                noise_multiplier=cfg["noise"],
                max_grad_norm=1.0,
                use_sc_dp=cfg["sc_dp"],
                use_swd=cfg["swd"],
                lambda_fair=0.5
            )
            
            report, _ = evaluate_model_pipeline(
                model=model,
                model_type=m_type,
                splits=splits,
                eps=eps,
                delta=delta,
                conformal_alpha=0.10,
                conformal_eps=1.0,
                use_group_conformal=cfg["group_cp"]
            )
            seed_metrics.append(report)
            if seed == 42:
                trained_models[m_type] = model
                
        # Aggregate across seeds
        agg_report = {"Dataset": dataset_name, "Model": m_type}
        metric_keys = [k for k in seed_metrics[0].keys() if k not in ["Model", "Epsilon", "Delta"]]
        for k in metric_keys:
            vals = [sm[k] for sm in seed_metrics]
            agg_report[f"{k}_mean"] = float(np.mean(vals))
            agg_report[f"{k}_std"] = float(np.std(vals))
        agg_report["Epsilon"] = seed_metrics[0]["Epsilon"]
        agg_report["Delta"] = seed_metrics[0]["Delta"]
        
        print(f"      AUC: {agg_report['AUC_mean']:.3f} +/- {agg_report['AUC_std']:.3f} | "
              f"ECE: {agg_report['ECE_mean']:.3f} | "
              f"DP Disp: {agg_report['DP_Disparity_mean']:.3f} | "
              f"Cov Disp: {agg_report['Coverage_Disparity_mean']:.3f} | "
              f"MIA AUC: {agg_report['MIA_AUC_mean']:.3f}")
        results.append(agg_report)
        
    return results, trained_models, last_splits, feature_names

def generate_publication_figures(all_results_df, trained_models_ehr, splits_ehr, feature_names_ehr, out_dir="figures"):
    os.makedirs(out_dir, exist_ok=True)
    print("\nGenerating Publication-Quality Visualizations...")
    
    # -------------------------------------------------------------
    # Figure 1: Clinical Utility (AUC) vs Algorithmic Fairness Disparity across Datasets
    # -------------------------------------------------------------
    fig, axes = plt.subplots(1, 3, figsize=(16, 5), dpi=300)
    datasets = all_results_df["Dataset"].unique()
    
    for ax, d_name in zip(axes, datasets):
        df_d = all_results_df[all_results_df["Dataset"] == d_name]
        
        # Plot bubble/scatter of AUC vs DP Disparity
        palette = {
            "Standard ERM": "#7f7f7f",
            "Standard DP-SGD": "#d62728",
            "Fair-ERM": "#ff7f0e",
            "Standard EDL": "#2ca02c",
            "DP-EDL": "#9467bd",
            "TRACER-Med (Proposed)": "#1f77b4",
            "TRACER-Med w/o SC-DP": "#8c564b",
            "TRACER-Med w/o SWD": "#e377c2",
            "TRACER-Med w/o DP-GCCP": "#bcbd22"
        }
        
        for _, row in df_d.iterrows():
            m = row["Model"]
            x_val = row["DP_Disparity_mean"]
            y_val = row["AUC_mean"]
            x_err = row["DP_Disparity_std"]
            y_err = row["AUC_std"]
            col = palette.get(m, "#333333")
            marker = "*" if "TRACER-Med (Proposed)" in m else ("o" if "DP" in m else "s")
            sz = 140 if "TRACER-Med (Proposed)" in m else 70
            
            ax.errorbar(x_val, y_val, xerr=x_err, yerr=y_err, fmt=marker, color=col, 
                        markersize=10 if "*" in marker else 7, capsize=3, label=m if d_name == datasets[0] else None)
            
        ax.set_title(d_name.split("(")[0].strip(), fontweight='bold')
        ax.set_xlabel("Demographic Parity Disparity (Lower is Fairer)")
        ax.set_ylabel("ROC-AUC (Higher is Better)")
        ax.set_xlim(left=-0.01)
        ax.axvspan(-0.01, 0.05, color='green', alpha=0.08, label='Clinical Equity Zone' if d_name == datasets[0] else None)

    axes[0].legend(loc='lower left', bbox_to_anchor=(0.0, -0.65), ncol=3, frameon=True)
    plt.tight_layout()
    fig1_path = os.path.join(out_dir, "fig1_utility_vs_fairness.pdf")
    fig1_png = os.path.join(out_dir, "fig1_utility_vs_fairness.png")
    fig.savefig(fig1_path, bbox_inches='tight')
    fig.savefig(fig1_png, bbox_inches='tight')
    plt.close(fig)
    print(f"Saved Figure 1 to {fig1_path} and {fig1_png}")
    
    # -------------------------------------------------------------
    # Figure 2: Conformal Prediction Subgroup Coverage Disparity
    # -------------------------------------------------------------
    fig, ax = plt.subplots(figsize=(9, 5), dpi=300)
    ehr_df = all_results_df[all_results_df["Dataset"].str.contains("EHR")]
    
    models = ehr_df["Model"].values
    coverage = ehr_df["Coverage_Marginal_mean"].values
    disparity = ehr_df["Coverage_Disparity_mean"].values
    
    x_indices = np.arange(len(models))
    width = 0.38
    
    rects1 = ax.bar(x_indices - width/2, coverage, width, label='Marginal Empirical Coverage', color='#3470a3', alpha=0.9)
    rects2 = ax.bar(x_indices + width/2, disparity, width, label='Subgroup Coverage Gap (Disparity)', color='#e05a47', alpha=0.9)
    
    ax.axhline(0.90, color='black', linestyle='--', linewidth=1.5, label='Nominal Target Coverage (1 - α = 90%)')
    ax.set_ylabel('Empirical Coverage / Disparity Ratio')
    ax.set_title('Conformal Calibration & Subgroup Equity on Clinical EHR Cohort', fontweight='bold')
    ax.set_xticks(x_indices)
    ax.set_xticklabels(models, rotation=35, ha='right')
    ax.set_ylim(0.0, 1.05)
    ax.legend(loc='upper right', frameon=True)
    
    plt.tight_layout()
    fig2_path = os.path.join(out_dir, "fig2_conformal_subgroup_coverage.pdf")
    fig2_png = os.path.join(out_dir, "fig2_conformal_subgroup_coverage.png")
    fig.savefig(fig2_path, bbox_inches='tight')
    fig.savefig(fig2_png, bbox_inches='tight')
    plt.close(fig)
    print(f"Saved Figure 2 to {fig2_path} and {fig2_png}")
    
    # -------------------------------------------------------------
    # Figure 3: Dual-Channel Dirichlet Attribution (DCDA) Heatmap
    # -------------------------------------------------------------
    tracer_model = trained_models_ehr.get("TRACER-Med (Proposed)", None)
    if tracer_model is not None:
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6), dpi=300)
        explainer = DualChannelDirichletExplainer(tracer_model, steps=40)
        
        # Pick 6 illustrative patients with positive diagnosis
        x_sample = splits_ehr["X_test"][:6].to("cuda" if torch.cuda.is_available() else "cpu")
        diag_attr, unc_attr = explainer.attribute(x_sample, target_class=1)
        
        feat_labels = feature_names_ehr if len(feature_names_ehr) == diag_attr.shape[1] else [f"Feat {i}" for i in range(diag_attr.shape[1])]
        patient_labels = [f"Patient #{i+1} (Deterioration)" for i in range(6)]
        
        sns.heatmap(diag_attr, annot=True, fmt=".2f", cmap="coolwarm", cbar=True,
                    xticklabels=feat_labels, yticklabels=patient_labels, ax=ax1, center=0)
        ax1.set_title("Channel A: Diagnostic Evidence Attribution\n(Features Driving Positive Disease Risk)", fontweight='bold')
        ax1.set_xticklabels(feat_labels, rotation=45, ha='right')
        
        sns.heatmap(unc_attr, annot=True, fmt=".2f", cmap="Blues", cbar=True,
                    xticklabels=feat_labels, yticklabels=patient_labels, ax=ax2)
        ax2.set_title("Channel B: Epistemic Uncertainty Attribution\n(Features Driving Clinical Doubt / Vacuity)", fontweight='bold')
        ax2.set_xticklabels(feat_labels, rotation=45, ha='right')
        
        plt.tight_layout()
        fig3_path = os.path.join(out_dir, "fig3_dual_channel_attribution.pdf")
        fig3_png = os.path.join(out_dir, "fig3_dual_channel_attribution.png")
        fig.savefig(fig3_path, bbox_inches='tight')
        fig.savefig(fig3_png, bbox_inches='tight')
        plt.close(fig)
        print(f"Saved Figure 3 to {fig3_path} and {fig3_png}")

    # -------------------------------------------------------------
    # Figure 4: Trade-off Radar Chart / Trustworthy Multi-Objective Balance
    # -------------------------------------------------------------
    fig, ax = plt.subplots(figsize=(8, 5), dpi=300)
    # Plot Calibration (1 - ECE), Fairness (1 - DP Disp), Privacy (1 - MIA AUC), Utility (AUC), Conformal (1 - Cov Disp)
    metrics_to_plot = ["AUC_mean", "ECE_mean", "DP_Disparity_mean", "MIA_AUC_mean", "Coverage_Disparity_mean"]
    labels = ["AUC (Utility)", "ECE (Calibration)", "DP Disp (Fairness)", "MIA AUC (Privacy Leakage)", "Coverage Gap"]
    
    comp_models = ["Standard ERM", "Standard DP-SGD", "Fair-ERM", "TRACER-Med (Proposed)"]
    df_sub = ehr_df[ehr_df["Model"].isin(comp_models)]
    
    x = np.arange(len(labels))
    width = 0.2
    
    for i, m in enumerate(comp_models):
        m_row = df_sub[df_sub["Model"] == m]
        if not m_row.empty:
            vals = [
                m_row["AUC_mean"].values[0],
                m_row["ECE_mean"].values[0],
                m_row["DP_Disparity_mean"].values[0],
                m_row["MIA_AUC_mean"].values[0],
                m_row["Coverage_Disparity_mean"].values[0]
            ]
            ax.bar(x + i*width, vals, width, label=m)
            
    ax.set_xticks(x + width*1.5)
    ax.set_xticklabels(labels, rotation=20, ha='right')
    ax.set_ylabel('Metric Value')
    ax.set_title('Trustworthiness Profile Comparison Across Core Clinical Pillars', fontweight='bold')
    ax.legend(loc='upper right', frameon=True)
    
    plt.tight_layout()
    fig4_path = os.path.join(out_dir, "fig4_trustworthiness_profile.pdf")
    fig4_png = os.path.join(out_dir, "fig4_trustworthiness_profile.png")
    fig.savefig(fig4_path, bbox_inches='tight')
    fig.savefig(fig4_png, bbox_inches='tight')
    plt.close(fig)
    print(f"Saved Figure 4 to {fig4_path} and {fig4_png}")

def main():
    start_time = time.time()
    all_results = []
    
    # 1. Statlog Heart
    X1, y1, s1, f1, name1 = load_statlog_heart()
    res1, _, _, _ = run_dataset_benchmark(name1, X1, y1, s1, f1, n_seeds=3)
    all_results.extend(res1)
    
    # 2. Pima Diabetes
    X2, y2, s2, f2, name2 = load_pima_diabetes()
    res2, _, _, _ = run_dataset_benchmark(name2, X2, y2, s2, f2, n_seeds=3)
    all_results.extend(res2)
    
    # 3. Multi-Center Clinical EHR Cohort
    X3, y3, s3, f3, name3 = load_clinical_ehr_cohort()
    res3, models3, splits3, feats3 = run_dataset_benchmark(name3, X3, y3, s3, f3, n_seeds=3)
    all_results.extend(res3)
    
    # Save combined results
    df_results = pd.DataFrame(all_results)
    out_csv = "experiments/results/benchmark_results.csv"
    out_json = "experiments/results/benchmark_results.json"
    df_results.to_csv(out_csv, index=False)
    with open(out_json, "w") as f:
        json.dump(all_results, f, indent=2)
    print(f"\nSaved all experimental benchmark metrics to {out_csv} and {out_json}")
    
    # Generate publication figures
    generate_publication_figures(df_results, models3, splits3, feats3, out_dir="figures")
    
    total_min = (time.time() - start_time) / 60.0
    print(f"\n=======================================================")
    print(f"ALL EXPERIMENTS AND VISUALIZATIONS COMPLETED in {total_min:.2f} minutes!")
    print(f"=======================================================")

if __name__ == "__main__":
    main()
