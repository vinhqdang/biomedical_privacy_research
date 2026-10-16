import json
import pandas as pd

with open('experiments/results/benchmark_results.json') as f:
    data = json.load(f)

df = pd.DataFrame(data)

print("\n=== FORMATTED TABLE 1 ENTRIES ===")
for ds in df['Dataset'].unique():
    print(f"\n--- {ds} ---")
    sub = df[df['Dataset'] == ds]
    for _, r in sub.iterrows():
        m = r['Model']
        auc = f"{r['AUC_mean']:.3f} \pm {r['AUC_std']:.2f}"
        ece = f"{r['ECE_mean']:.3f} \pm {r['ECE_std']:.2f}"
        brier = f"{r['Brier_mean']:.3f} \pm {r['Brier_std']:.3f}"
        sp = f"{r['DP_Disparity_mean']:.3f} \pm {r['DP_Disparity_std']:.2f}"
        eo = f"{r['EO_Disparity_mean']:.3f} \pm {r['EO_Disparity_std']:.2f}"
        mia = f"{r['MIA_AUC_mean']:.3f} \pm {r['MIA_AUC_std']:.2f}"
        cov = f"{r['Coverage_Disparity_mean']:.3f} \pm {r['Coverage_Disparity_std']:.3f}"
        infd = f"{r['Infidelity_mean']:.3f} \pm {r['Infidelity_std']:.3f}"
        print(f"{m:25s} & ${auc}$ & ${ece}$ & ${brier}$ & ${sp}$ & ${eo}$ & ${mia}$ & ${cov}$ & ${infd}$ \\\\")

print("\n=== FORMATTED TABLE 2 (ABLATION ON HEART) ===")
sub_heart = df[df['Dataset'].str.contains('Heart')]
ablation_models = [
    "TRACER-Med (Proposed)",
    "TRACER-Med w/o SC-DP",
    "TRACER-Med w/o SWD",
    "TRACER-Med w/o DP-GCCP"
]
for m in ablation_models:
    r = sub_heart[sub_heart['Model'] == m].iloc[0]
    auc = f"{r['AUC_mean']:.3f} \pm {r['AUC_std']:.2f}"
    ece = f"{r['ECE_mean']:.3f} \pm {r['ECE_std']:.2f}"
    sp = f"{r['DP_Disparity_mean']:.3f} \pm {r['DP_Disparity_std']:.2f}"
    eo = f"{r['EO_Disparity_mean']:.3f} \pm {r['EO_Disparity_std']:.2f}"
    cov = f"{r['Coverage_Disparity_mean']:.3f} \pm {r['Coverage_Disparity_std']:.3f}"
    print(f"{m:25s} & ${auc}$ & ${ece}$ & ${sp}$ & ${eo}$ & ${cov}$ \\\\")
