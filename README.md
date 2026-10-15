# TRACER-Med: Harmonizing Privacy, Fairness, and Uncertainty in Clinical Deep Learning

**Author:** Quang-Vinh Dang  
**Affiliation:** British University Vietnam, Hung Yen, Vietnam  
**Contact:** `vinh.dq4@buv.edu.vn`  
**Target Venue:** Elsevier (*Artificial Intelligence in Medicine*)  

---

## Overview

**TRACER-Med** (**TR**ustworthy, **A**daptive, **C**onformalized, **E**vidential, and Differentially P**R**ivate Deep Learning for **Med**icine) resolves the **Privacy--Fairness--Uncertainty Trilemma** in clinical machine learning by unifying:

1. **Evidential Dirichlet Neural Networks:** Second-order evidential formulation parameterizing Dirichlet beliefs to analytically decouple aleatoric entropy from epistemic vacuity.
2. **Subgroup-Calibrated DP-SGD (SC-DP-SGD):** Group-conditioned clipping and dynamic sensitivity reweighting strictly bounding global sensitivity by $C_{\max}$ under R\'enyi Differential Privacy, eliminating the disparate privacy penalty on minority patient cohorts.
3. **Latent Sliced Wasserstein Demographic Alignment:** Pre-training/regularization module aligning penultimate representations across demographic cohorts without cross-sample gradient coupling during private optimization.
4. **Differentially Private Group-Conditional Conformal Prediction (DP-GCCP):** Private empirical quantile mechanism providing finite-sample subgroup coverage guarantees ($\ge 1 - \alpha_{\text{conf}}$) with split-level privacy accounting.
5. **Dual-Channel Dirichlet Attribution (DCDA):** Closed-form path-integrated feature attributions simultaneously explaining diagnostic risk evidence (Channel A) and clinical doubt/vacuity (Channel B).

---

## Repository Structure

```
biomedical_privacy_research/
├── manuscript/                  # Elsevier LaTeX manuscript & compiled PDF
│   ├── main.tex                 # Manuscript source (elsarticle format)
│   ├── references.bib           # Complete bibliography with DOIs (up to Oct-2026)
│   ├── main.pdf                 # Compiled 26-page publication-ready PDF
│   ├── main.bbl                 # Pre-compiled bibliography
│   └── figures/                 # Publication vector PDF and PNG figures
│       ├── fig1_utility_vs_fairness.pdf
│       ├── fig2_conformal_subgroup_coverage.pdf
│       ├── fig3_dual_channel_attribution.pdf
│       └── fig4_trustworthiness_profile.pdf
├── src/                         # PyTorch core algorithm implementation
│   ├── edl_model.py             # Evidential Dirichlet neural network
│   ├── dp_optimizer.py          # Subgroup-Calibrated DP-SGD engine
│   ├── fairness.py              # Sliced Wasserstein Distance alignment
│   ├── conformal.py             # Differentially Private Group-Conditional CP
│   ├── xai.py                   # Dual-Channel Dirichlet Attribution (DCDA)
│   ├── metrics.py               # Comprehensive 6-pillar clinical evaluation suite
│   └── train_eval.py            # Training and cross-validation pipelines
├── data/                        # Clinical benchmark datasets & loader
│   ├── heart_statlog.csv        # Statlog Heart Disease benchmark (N=270)
│   ├── pima_diabetes.csv        # Pima Indians Diabetes benchmark (N=768)
│   ├── clinical_ehr_cohort.csv  # Synthetic Inpatient EHR Cohort (N=10,000)
│   └── download_benchmarks.py   # Benchmark preparation and verification script
├── experiments/                 # Benchmark experiment orchestration
│   ├── run_all_experiments.py   # Full benchmark execution and figure generation
│   └── results/                 # Structured CSV and JSON benchmark results
├── figures/                     # Root-level publication figures
├── review/                      # Automated peer review reports and referee scripts
│   ├── run_claude_review.py     # Claude review runner script
│   ├── review_round1.md         # Initial comprehensive review report
│   ├── review_round2.md         # Major revision audit report
│   ├── review_round3.md         # Third-round verification report
│   └── review_round4.md         # Final pre-submission audit report
├── LICENSE                      # BSD-3-Clause License
└── README.md                    # Repository documentation
```

---

## Compiling the Manuscript

The manuscript uses Elsevier's `elsarticle` document class and compiles cleanly with MiKTeX or TeX Live:

```bash
cd manuscript
pdflatex -interaction=nonstopmode main.tex
bibtex main
pdflatex -interaction=nonstopmode main.tex
pdflatex -interaction=nonstopmode main.tex
```

The resulting [`main.pdf`](manuscript/main.pdf) contains all 4 high-resolution figures, full mathematical derivations, algorithmic pseudocode, and hyperlinked references with valid DOIs.

---

## Running Experiments

Activate your Python 3.13 / PyTorch environment (CUDA GPU recommended):

```bash
conda activate py313
python experiments/run_all_experiments.py
```

This runs stratified 3-fold cross-validation across all three benchmarks and models (Standard ERM, Standard DP-SGD, Fair-ERM, Standard EDL, DP-EDL, and TRACER-Med), generating:
- Tabular performance metrics (`experiments/results/benchmark_results.csv`)
- High-resolution figures (`figures/` and `manuscript/figures/`)

---

## License

This project is licensed under the BSD-3-Clause License. See [`LICENSE`](LICENSE) for details.