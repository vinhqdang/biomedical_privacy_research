"""Builds the LaTeX result tables of the manuscript from the CSV files in experiments/results/."""
import pandas as pd

d = pd.read_csv("experiments/results/benchmark_results.csv")
DS = list(dict.fromkeys(d["Dataset"]))
TITLES = ["Statlog Heart Disease ($N=270$, protected: sex)",
          "Pima Indians Diabetes ($N=768$, protected: age group)",
          "Diabetes 130-US Hospitals, real ($N=20{,}000$, 30-day readmission, protected: race)",
          "Simulated inpatient EHR cohort ($N=10{,}000$, protected: demographic cohort)"]
ORDER = ["Standard ERM", "Fair-ERM", "Standard EDL", "Standard DP-SGD", "DPSGD-F", "PF-LD (adapted, matched eps)",
         "PF-LD (adapted)", "DP-EDL", "TRACER-Med (Proposed)"]
LAB = {"Standard DP-SGD": "DP-SGD", "DPSGD-F": "DPSGD-F~\\cite{xu2021removing}",
       "PF-LD (adapted, matched eps)": "PF-LD, adapted, matched $\\epsilon$~\\cite{tran2021differentially}",
       "PF-LD (adapted)": "PF-LD, adapted, $\\sigma=0.8$~\\cite{tran2021differentially}",
       "TRACER-Med (Proposed)": "\\textbf{TRACER-Med}"}
BOLD_AMONG = ["Standard DP-SGD", "DPSGD-F", "PF-LD (adapted, matched eps)", "DP-EDL", "TRACER-Med (Proposed)"]
COLS = [("AUC", "up", 3), ("ECE", "dn", 3), ("DP_Disparity", "dn", 3), ("EO_Disparity", "dn", 3), ("MIA_AUC", "dn", 3)]

def fmt(r, m, p, bold):
    s = f"{r[m+'_mean']:.{p}f} \\pm {r[m+'_std']:.{p}f}"
    if bold:
        s = "\\mathbf{" + s.replace(" \\pm ", "} \\pm \\mathbf{") + "}"
    return "$" + s + "$"

def eps_cell(e):
    return "$\\infty$" if e == float("inf") or e > 1e6 else f"{e:.1f}"

lines = []
for i, ds in enumerate(DS):
    g = d[d.Dataset == ds]
    if i:
        lines.append("\\midrule")
    lines.append("\\multicolumn{9}{l}{\\emph{" + TITLES[i] + "}} \\\\")
    best = {}
    for m, dirn, _ in COLS:
        v = g[g.Model.isin(BOLD_AMONG)][m + "_mean"]
        best[m] = v.max() if dirn == "up" else v.min()
    for mod in ORDER:
        r = g[g.Model == mod].iloc[0]
        cells = [eps_cell(r["Epsilon"])]
        for m, dirn, p in COLS:
            cells.append(fmt(r, m, p, mod in BOLD_AMONG and abs(r[m + "_mean"] - best[m]) < 1e-12))
        cov = f"${r['Coverage_Marginal_mean']:.2f}$"
        if r["Coverage_Marginal_mean"] < 0.85:
            cov = f"${r['Coverage_Marginal_mean']:.2f}^\\dagger$"
        cells.append(cov)
        cells.append(f"${r['Avg_Set_Size_mean']:.2f}$")
        lines.append(LAB.get(mod, mod) + " & " + " & ".join(cells) + " \\\\")
open("manuscript/table_main_body.tex", "w").write("\n".join(lines) + "\n")

ABL = ["TRACER-Med (Proposed)", "TRACER-Med w/o SC-DP", "TRACER-Med w/o SWD", "TRACER-Med w/o DP-GCCP"]
AL = {"TRACER-Med (Proposed)": "\\textbf{TRACER-Med (full)}", "TRACER-Med w/o SC-DP": "\\quad w/o REF-DP (uniform clipping)",
      "TRACER-Med w/o SWD": "\\quad w/o PC-PAT (no prototype alignment)", "TRACER-Med w/o DP-GCCP": "\\quad w/o group-conditional conformal"}
acols = [("AUC", "up", 3), ("ECE", "dn", 3), ("DP_Disparity", "dn", 3), ("EO_Disparity", "dn", 3)]
lines = []
for i, ds in enumerate(DS):
    g = d[d.Dataset == ds]
    if i:
        lines.append("\\midrule")
    lines.append("\\multicolumn{8}{l}{\\emph{" + TITLES[i].split(" (")[0] + "}} \\\\")
    best = {m: (g[g.Model.isin(ABL)][m + "_mean"].max() if dn == "up" else g[g.Model.isin(ABL)][m + "_mean"].min()) for m, dn, _ in acols}
    for mod in ABL:
        r = g[g.Model == mod].iloc[0]
        cs = [eps_cell(r["Epsilon"])] + [fmt(r, m, p, abs(r[m + "_mean"] - best[m]) < 1e-12) for m, _, p in acols]
        cs.append(f"${r['Coverage_Marginal_mean']:.2f}$")
        cs.append(f"${r['Avg_Set_Size_mean']:.2f}$")
        lines.append(AL[mod] + " & " + " & ".join(cs) + " \\\\")
open("manuscript/table_ablation_body.tex", "w").write("\n".join(lines) + "\n")

sw = pd.read_csv("experiments/results/noise_sweep.csv")
MS = ["Standard DP-SGD", "DPSGD-F", "PF-LD (adapted)", "DP-EDL", "TRACER-Med (Proposed)"]
lines = []
for i, ds in enumerate(dict.fromkeys(sw["Dataset"])):
    g = sw[sw.Dataset == ds]
    if i:
        lines.append("\\midrule")
    lines.append("\\multicolumn{12}{l}{\\emph{" + TITLES[DS.index(ds)].split(" (")[0] + "}} \\\\")
    for sg, gg in g.groupby("Sigma"):
        eps = gg[gg.Model == "Standard DP-SGD"]["Epsilon"].iloc[0]
        cs = []
        for m in ["AUC", "EO_Disparity"]:
            for mod in MS:
                cs.append(f"{gg[gg.Model == mod].iloc[0][m + '_mean']:.3f}")
        lines.append(f"{sg:.1f} & {eps:.1f} & " + " & ".join(cs) + " \\\\")
open("manuscript/table_sweep_body.tex", "w").write("\n".join(lines) + "\n")
