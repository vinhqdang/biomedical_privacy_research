"""Builds the LaTeX result tables of the manuscript from experiments/results/benchmark_results.csv."""
import pandas as pd, numpy as np
d = pd.read_csv("experiments/results/benchmark_results.csv")
M = [("AUC", "up", 3), ("ECE", "dn", 3), ("Brier", "dn", 3), ("DP_Disparity", "dn", 3), ("EO_Disparity", "dn", 3),
     ("MIA_AUC", "dn", 3), ("Coverage_Disparity", "dn", 3), ("Avg_Set_Size", "dn", 2), ("Infidelity", "dn", 3)]
ORDER = ["Standard ERM", "Standard DP-SGD", "Fair-ERM", "DPSGD-F (simplified)", "Standard EDL", "DP-EDL", "TRACER-Med (Proposed)"]
PRIV = ["Standard DP-SGD", "DPSGD-F (simplified)", "DP-EDL", "TRACER-Med (Proposed)"]
LAB = {"Standard DP-SGD": "Standard DP-SGD", "DPSGD-F (simplified)": "DPSGD-F (simplified)~\\cite{xu2021removing}",
       "TRACER-Med (Proposed)": "\\textbf{TRACER-Med (Proposed)}"}
TITLES = ["Benchmark 1: Statlog Heart Disease ($N=270, D=12$, Protected: Biological Sex)",
          "Benchmark 2: Pima Indians Diabetes ($N=768, D=8$, Protected: Age Category)",
          "Benchmark 3: Simulated Inpatient EHR Cohort ($N=10{,}000, D=15$, Protected: Demographic Cohort)"]
def vac(r): return r["Avg_Set_Size_mean"] >= 1.99
def cell(r, m, bold, star):
    s = f"{r[m+'_mean']:.{dict((a,c) for a,_,c in M)[m]}f} \\pm {r[m+'_std']:.{dict((a,c) for a,_,c in M)[m]}f}"
    if bold: s = "\\mathbf{" + s.replace(" \\pm ", "} \\pm \\mathbf{") + "}"
    return "$" + s + ("^*" if star else "") + "$"
def block(rows, models, bold_among):
    out = []
    best = {}
    for m, dirn, _ in M:
        cand = rows[rows.Model.isin(bold_among)]
        if m in ("Coverage_Disparity", "Avg_Set_Size"): cand = cand[cand.Avg_Set_Size_mean < 1.99]
        if cand.empty: continue
        v = cand[m+"_mean"]; best[m] = v.max() if dirn == "up" else v.min()
    for mod in models:
        r = rows[rows.Model == mod].iloc[0]
        cells = []
        for m, dirn, _ in M:
            star = vac(r) and m in ("Coverage_Disparity", "Avg_Set_Size")
            b = (mod in bold_among) and m in best and abs(r[m+"_mean"] - best[m]) < 1e-12 and not star
            cells.append(cell(r, m, b, star))
        out.append(LAB.get(mod, mod) + " & " + " & ".join(cells) + " \\\\")
    return out
lines = []
for i, (ds, g) in enumerate(d.groupby("Dataset", sort=False)):
    if i: lines.append("\\midrule")
    lines.append("\\multicolumn{10}{l}{\\emph{" + TITLES[i] + "}} \\\\")
    lines += block(g, ORDER, PRIV)
open("manuscript/table_main_body.tex", "w").write("\n".join(lines) + "\n")
ABL = ["TRACER-Med (Proposed)", "TRACER-Med w/o SC-DP", "TRACER-Med w/o SWD", "TRACER-Med w/o DP-GCCP"]
AL = {"TRACER-Med (Proposed)": "\\textbf{TRACER-Med (Full)}", "TRACER-Med w/o SC-DP": "\\quad w/o REF-DP (uniform clipping)",
      "TRACER-Med w/o SWD": "\\quad w/o PC-PAT (no prototype alignment)", "TRACER-Med w/o DP-GCCP": "\\quad w/o DP-GCCP (marginal CP)"}
cols = [("AUC","up",3),("ECE","dn",3),("DP_Disparity","dn",3),("EO_Disparity","dn",3),("Coverage_Disparity","dn",3),("Avg_Set_Size","dn",2)]
lines = []
for i, (ds, g) in enumerate(d.groupby("Dataset", sort=False)):
    if i: lines.append("\\midrule")
    lines.append("\\multicolumn{7}{l}{\\emph{" + TITLES[i].split(" (")[0] + "}} \\\\")
    for mod in ABL:
        r = g[g.Model == mod].iloc[0]
        cs = []
        for m, dirn, p in cols:
            vals = g[g.Model.isin(ABL)]
            if m in ("Coverage_Disparity",): vals = vals[vals.Avg_Set_Size_mean < 1.99]
            best = vals[m+"_mean"].max() if dirn == "up" else vals[m+"_mean"].min()
            star = vac(r) and m in ("Coverage_Disparity", "Avg_Set_Size")
            bold = (not star) and abs(r[m+"_mean"]-best) < 1e-12 and m != "Avg_Set_Size"
            s = f"{r[m+'_mean']:.{p}f} \\pm {r[m+'_std']:.{p}f}"
            if bold: s = "\\mathbf{" + s.replace(" \\pm ", "} \\pm \\mathbf{") + "}"
            cs.append("$" + s + ("^*" if star else "") + "$")
        lines.append(AL[mod] + " & " + " & ".join(cs) + " \\\\")
open("manuscript/table_ablation_body.tex", "w").write("\n".join(lines) + "\n")

# Noise-sweep table (experiments/results/noise_sweep.csv)
sw = pd.read_csv("experiments/results/noise_sweep.csv")
MS = ["Standard DP-SGD", "DP-EDL", "TRACER-Med (Proposed)"]
lines = []
for i, (ds, g) in enumerate(sw.groupby("Dataset", sort=False)):
    if i: lines.append("\\midrule")
    lines.append("\\multicolumn{11}{l}{\\emph{" + TITLES[i].split(" (")[0] + "}} \\\\")
    for sg, gg in g.groupby("Sigma"):
        eps = gg["Epsilon"].iloc[0]
        cs = []
        for m in ["AUC", "DP_Disparity", "ECE"]:
            for mod in MS:
                r = gg[gg.Model == mod].iloc[0]
                cs.append(f"{r[m+'_mean']:.3f}")
        lines.append(f"{sg:.1f} & {eps:.1f} & " + " & ".join(cs) + " \\\\")
open("manuscript/table_sweep_body.tex", "w").write("\n".join(lines) + "\n")
