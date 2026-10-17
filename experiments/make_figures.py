"""Builds the manuscript figures from experiments/results/benchmark_results.csv.
Figure 3 retrains the TRACER-Med model on the simulated cohort (split seed 42) to compute attributions."""
import os, sys
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
import numpy as np, pandas as pd, torch
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns

plt.rcParams.update({"font.size": 9, "axes.titlesize": 10, "axes.labelsize": 9, "legend.fontsize": 8,
                     "xtick.labelsize": 8, "ytick.labelsize": 8, "pdf.fonttype": 42, "ps.fonttype": 42})
STYLE = {  # model: (label, color, marker)
    "Standard ERM": ("ERM", "#7f7f7f", "s"), "Fair-ERM": ("Fair-ERM", "#ff7f0e", "s"), "Standard EDL": ("EDL", "#2ca02c", "s"),
    "Standard DP-SGD": ("DP-SGD", "#d62728", "o"), "DPSGD-F": ("DPSGD-F", "#000000", "o"),
    "PF-LD (adapted, matched eps)": ("PF-LD, matched $\\epsilon$", "#8c564b", "^"), "PF-LD (adapted)": ("PF-LD, $\\sigma$=0.8", "#bc8f8f", "^"),
    "DP-EDL": ("DP-EDL", "#9467bd", "o"), "TRACER-Med (Proposed)": ("TRACER-Med", "#1f77b4", "*"),
}
SHORT = {"Statlog Heart (Cardiovascular)": "Statlog Heart (n=270)", "Pima Diabetes (Endocrine/Metabolic)": "Pima Diabetes (n=768)",
         "Diabetes 130-US Hospitals (Readmission, real)": "Diabetes 130-US Hospitals, real (n=20000)",
         "Multi-Center Clinical EHR Cohort (Acuity/Mortality)": "Simulated EHR cohort (n=10000)"}

def fig1(d, outs):
    fig, axes = plt.subplots(2, 2, figsize=(7.2, 6.4), dpi=300)
    for ax, ds in zip(axes.ravel(), SHORT):
        g = d[d.Dataset == ds]
        for m, (lab, col, mk) in STYLE.items():
            r = g[g.Model == m].iloc[0]
            ax.errorbar(r.EO_Disparity_mean, r.AUC_mean, xerr=r.EO_Disparity_std, yerr=r.AUC_std, fmt=mk, color=col,
                        markersize=9 if mk == "*" else 5, capsize=2, elinewidth=0.8, label=lab)
        ax.set_title(SHORT[ds]); ax.set_xlabel("Equalized-odds disparity (lower is fairer)"); ax.set_ylabel("ROC-AUC")
        ax.grid(alpha=0.3)
    h, l = axes[0, 0].get_legend_handles_labels()
    fig.legend(h, l, loc="lower center", ncol=5, frameon=False)
    fig.tight_layout(rect=(0, 0.08, 1, 1))
    for o in outs:
        fig.savefig(os.path.join(o, "fig1_utility_vs_fairness.pdf")); fig.savefig(os.path.join(o, "fig1_utility_vs_fairness.png"))
    plt.close(fig)

def fig2(d, outs):
    g = d[d.Dataset == "Multi-Center Clinical EHR Cohort (Acuity/Mortality)"]
    models = [m for m in STYLE if m in set(g.Model)]
    x = np.arange(len(models)); w = 0.38
    fig, ax = plt.subplots(figsize=(7.2, 3.8), dpi=300)
    cov = [g[g.Model == m].Coverage_Marginal_mean.iloc[0] for m in models]
    gap = [g[g.Model == m].Coverage_Disparity_mean.iloc[0] for m in models]
    ax.bar(x - w / 2, cov, w, color="#3470a3", label="Marginal coverage")
    ax.bar(x + w / 2, gap, w, color="#e05a47", label="Coverage gap between the two groups")
    ax.axhline(0.90, color="black", ls="--", lw=1.2, label="Nominal coverage 0.90")
    ax.set_xticks(x); ax.set_xticklabels([STYLE[m][0] for m in models], rotation=30, ha="right")
    ax.set_ylabel("Coverage"); ax.set_ylim(0, 1.18); ax.legend(loc="upper center", ncol=3, frameon=False)
    fig.tight_layout()
    for o in outs:
        fig.savefig(os.path.join(o, "fig2_conformal_subgroup_coverage.pdf")); fig.savefig(os.path.join(o, "fig2_conformal_subgroup_coverage.png"))
    plt.close(fig)

def fig4(d, outs):
    models = ["Standard DP-SGD", "DPSGD-F", "PF-LD (adapted, matched eps)", "DP-EDL", "TRACER-Med (Proposed)"]
    mets = [("AUC_mean", "ROC-AUC (higher is better)"), ("ECE_mean", "Expected calibration error (lower)"),
            ("EO_Disparity_mean", "Equalized-odds disparity (lower)"), ("MIA_AUC_mean", "Membership-inference AUC (lower)")]
    fig, axes = plt.subplots(2, 2, figsize=(7.2, 6.0), dpi=300)
    dss = list(SHORT); x = np.arange(len(dss)); w = 0.16
    for ax, (m, title) in zip(axes.ravel(), mets):
        for i, mod in enumerate(models):
            vals = [d[(d.Dataset == ds) & (d.Model == mod)][m].iloc[0] for ds in dss]
            ax.bar(x + (i - 2) * w, vals, w, color=STYLE[mod][1], label=STYLE[mod][0])
        ax.set_xticks(x); ax.set_xticklabels(["Heart", "Pima", "Readmission\n(real)", "Simulated\nEHR"]); ax.set_title(title)
        if m == "MIA_AUC_mean": ax.set_ylim(0.45, 0.6)
        ax.grid(axis="y", alpha=0.3)
    h, l = axes[0, 0].get_legend_handles_labels()
    fig.legend(h, l, loc="lower center", ncol=5, frameon=False)
    fig.tight_layout(rect=(0, 0.06, 1, 1))
    for o in outs:
        fig.savefig(os.path.join(o, "fig4_trustworthiness_profile.pdf")); fig.savefig(os.path.join(o, "fig4_trustworthiness_profile.png"))
    plt.close(fig)

def fig3(outs):
    from data.download_benchmarks import load_clinical_ehr_cohort
    from src.train_eval import prepare_dataset_splits, train_model
    from src.xai import DualChannelDirichletExplainer
    X, y, s, feats, _ = load_clinical_ehr_cohort()
    sp = prepare_dataset_splits(X, y, s, seed=42)
    torch.manual_seed(42); np.random.seed(42)
    model, _, _ = train_model("TRACER-Med (Proposed)", sp, epochs=12, lr=0.005, batch_size=128, noise_multiplier=0.8,
                              use_sc_dp=True, use_swd=True, lambda_fair=0.5)
    pos = np.where(sp["y_test"].numpy() == 1)[0][:3]; neg = np.where(sp["y_test"].numpy() == 0)[0][:3]
    idx = np.concatenate([pos, neg])
    xs = sp["X_test"][idx]
    a_risk, a_doubt = DualChannelDirichletExplainer(model, steps=30).attribute(xs, target_class=1)
    labels = [f"Patient {i+1} (observed outcome {int(sp['y_test'][j])})" for i, j in enumerate(idx)]
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(7.4, 3.6), dpi=300)
    sns.heatmap(a_risk, annot=True, fmt=".2f", cmap="coolwarm", center=0, xticklabels=feats, yticklabels=labels, ax=ax1,
                annot_kws={"size": 5}, cbar_kws={"shrink": 0.8})
    ax1.set_title("Channel A: disease risk")
    sns.heatmap(a_doubt, annot=True, fmt=".2f", cmap="coolwarm", center=0, xticklabels=feats, yticklabels=False, ax=ax2,
                annot_kws={"size": 5}, cbar_kws={"shrink": 0.8})
    ax2.set_title("Channel B: epistemic doubt")
    for ax in (ax1, ax2): ax.tick_params(axis="x", labelrotation=60, labelsize=6)
    fig.tight_layout()
    for o in outs:
        fig.savefig(os.path.join(o, "fig3_dual_channel_attribution.pdf")); fig.savefig(os.path.join(o, "fig3_dual_channel_attribution.png"))
    plt.close(fig)
    np.save("experiments/results/fig3_risk_attr.npy", a_risk); np.save("experiments/results/fig3_doubt_attr.npy", a_doubt)

def make_all(outs=("figures", "manuscript/figures")):
    for o in outs: os.makedirs(o, exist_ok=True)
    d = pd.read_csv("experiments/results/benchmark_results.csv")
    fig1(d, outs); fig2(d, outs); fig4(d, outs); fig3(outs)

if __name__ == "__main__":
    make_all()
