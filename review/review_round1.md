# Referee Report: "TRACER-Med: A Unified Trustworthy Architecture for Differentially Private, Subgroup-Fair, and Uncertainty-Calibrated Clinical Decision Support with Intrinsic Explainability"

I reviewed the LaTeX as submitted. I also read `data/download_benchmarks.py` in the repo to check what the "Multi-Center EHR Cohort" really is. That check changes the verdict.

---

## 1. Overall Assessment & Summary of Contributions

**Summary.** The manuscript proposes TRACER-Med, which combines four pieces:
1. An evidential (Dirichlet) classifier.
2. A group-calibrated DP-SGD variant (SC-DP-SGD), with a sliced-Wasserstein (SWD) latent-alignment penalty.
3. A per-group "private" conformal predictor (DP-GCCP).
4. Dual-channel integrated-gradients explanations (DCDA).

It claims:
- up to 74% of DP-induced disparity is eliminated;
- "exact nominal 90%" subgroup coverage;
- MIA AUC of 0.518;
- faithful dual explanations on three benchmarks.

**Assessment.** The topic is timely and the intended pipeline is sensible. In its present form, though, the paper cannot support its central claims, for four reasons:
- **The headline privacy theorem is incorrect.** The proof does not establish the stated guarantee, and the mechanism as specified likely violates it.
- **The DP conformal mechanism is mis-calibrated and its guarantee is overstated.**
- **The "real-world multi-center EHR" cohort is synthetic.** The repo generates it from independent Gaussian/lognormal/Poisson draws. Both the fairness and the clinical results on it are therefore artefacts of the generator.
- **Several reported numbers cannot be reproduced from the text.** Others contradict it.

Recommendation: **major revision, bordering on reject.** The idea is publishable if the theory is fixed and real data are used.

---

## 2. Major Strengths

1. **Well-motivated integration problem.** Privacy, fairness, calibration, coverage and explanation interact. A joint study is genuinely useful.
2. **Reasonable component choices.** EDL, subgroup-aware clipping, group-conditional conformal prediction and uncertainty-aware attribution are sensible building blocks.
3. **Informative ablation design.** Table 2 isolates each module. It also reports the coverage-gap metric separately from fairness, which is good practice.
4. **Useful metric breadth.** The paper reports ECE, Brier, ΔDP, ΔEO, MIA, coverage gap and infidelity together.
5. **Clinically motivated deployment narrative.** The singlet/abstain workflow and the Channel B idea are attractive.
6. **Good internal arithmetic in places.** The coverage figures are consistent: 0.71·95.6 + 0.29·74.2 = 89.4, and 0.71·91.4 + 0.29·88.3 ≈ 90.5. The 78% reduction (0.342 → 0.074) is also correct.
7. **Clear structure and generally readable exposition.**

---

## 3. Critical Methodological & Mathematical Issues

### 3.1 Theorem 1 (Privacy Invariance) is incorrect — **CRITICAL**

1. **Sensitivity exceeds the noise calibration.**
   - The clip threshold is $C_a = C\tilde w_a$. The noise is $\mathcal N(0,\sigma^2C^2I)$, calibrated to $C$.
   - The proof itself says the sensitivity is bounded by $C\max_a\tilde w_a$, and then claims *identical* guarantees. Those statements contradict each other.
   - Worked example for γ=1, minority fraction 0.29: $w_0=1.72$, $w_1=0.70$. The normaliser is $\sqrt{0.29\cdot1.72^2+0.71\cdot0.70^2}\approx1.10$. So $\tilde w_0\approx1.56$ and the true sensitivity is ≈1.56C.
   - The effective noise multiplier is then $\sigma/1.56$, which is strictly weaker privacy. The reported ε=2.45 is therefore wrong by construction.
2. **The normalisation is an average, not a bound.** "$\mathbb E[w_a^2]=1$" constrains a batch mean. DP sensitivity is a worst case over a single record, so the normalisation does nothing for sensitivity.
3. **The weights depend on the private data.**
   - $\tilde w_a$ is computed from batch group counts $n_0,n_1$.
   - Adding or removing one record changes the counts. That changes the clip bound of every *other* record in the batch.
   - The sum $\sum\bar g_i$ then has sensitivity well beyond one clipped gradient. The standard DP-SGD analysis does not apply.
4. **The sensitive attribute is itself private.** Group membership and counts must be protected or treated as public and justified.
5. **The RDP bound is not valid as stated.**
   - $\mathcal R_\alpha\le q^2\alpha T/(2\sigma^2)$ is not the subsampled-Gaussian RDP bound. The Gaussian mechanism alone gives $\alpha/(2\sigma^2)$ with no $q^2$.
   - The subsampled bound is roughly $2q^2\alpha/\sigma^2$ and holds only for restricted $(\alpha,\sigma,q)$. See Mironov et al. 2019, "Rényi Differential Privacy of the Sampled Gaussian Mechanism".
   - The RDP→(ε,δ) conversion is missing. For example, $\varepsilon=\mathcal R_\alpha+\log(1/\delta)/(\alpha-1)$, or the tighter Balle/Canonne conversions.
   - "(ε,δ)-Rényi DP" is a misnomer; RDP is $(\alpha,\varepsilon)$-RDP.
   - Poisson versus fixed-size batches and add/remove versus replace adjacency are never specified.
6. **A valid fix exists.** Either:
   - choose weights that depend only on public group priors, and normalise so that $\max_a\tilde w_a\le1$ (or scale the noise by $\sigma\max_a C_a$); or
   - state and prove the privacy cost of the data-dependent scheme.

   Existing work does this and should be compared against: Tran et al. 2021 (DP-SGD-F), Xu et al. 2021 ("Removing disparate impact on model accuracy in DP-SGD"), and Esipova et al. 2022 (gradient misalignment).

### 3.2 The SWD fairness term breaks per-example DP-SGD — **CRITICAL**

- $\mathcal L_{\text{fair}}$ is a distributional quantity over two groups within a batch. In Algorithm 1 it is written as a per-sample term, $\mathcal L_{\text{fair}}(\mathbf z_i,a_i)$. That is ill-defined.
- Because it couples examples within a batch, example $i$'s gradient depends on others. Per-example clipping no longer bounds the effect of one record.
- With roughly 10 minority points per batch on Statlog, a Monte-Carlo SWD with random projections is extremely noisy.
- The 1D-quantile expression is $W_1$ per direction. Say so explicitly. $\boldsymbol\theta_m$ should be called $\boldsymbol\omega_m\sim\mathrm{Unif}(\mathbb S^{d-1})$.
- **Demographic-parity-style alignment conflicts with equalized odds when base rates differ.** Statlog has very different CAD prevalence by sex. The paper optimises one criterion and reports the other as an improvement without discussion.
- **The Fair-ERM comparison undercuts the contribution.**
  - Non-private Fair-ERM (SWD only) already reaches ΔDP 0.112, 0.098 and 0.085.
  - TRACER-Med's gains therefore come from SWD, not from SC-DP.
  - The right control is **DP-SGD + SWD** (a private Fair-ERM). It is absent.

### 3.3 Evidential formulation

1. **Aleatoric uncertainty is never defined.** The paper promises "disentangling" but only defines vacuity $u=K/S$. Give the expected entropy, and the mutual information as the epistemic part. Vacuity is also a deterministic function of evidence $S$ and is not "epistemic ignorance" in a Bayesian sense.
2. **Known critiques are not discussed.**
   - Bengs et al. 2022 ("Pitfalls of epistemic UQ through loss minimization").
   - Shen et al. 2023 ("Are UQ capabilities of EDL a mirage?").
   - Evidence that EDL vacuity is not a faithful epistemic measure.
3. **KL annealing is missing.** $\lambda_{\text{reg}}$ is constant in the equation. Sensoy anneals it from 0. Specify it. Also check that $\tilde{\boldsymbol\alpha}$ and the KL are defined for $K=2$, $\mathbf y$ one-hot.
4. **Notation clash.** $\alpha$ means the Dirichlet parameter, the conformal miscoverage level, the RDP order and the DP ($\varepsilon,\delta$) setting. $\theta$ means both the model weights and the SWD projections. $\Delta_{\text{DP}}$ (demographic parity) collides with DP (differential privacy). $\mathcal D$ means both the dataset and the KL.
5. **Equation (3) as written.** $\mathbb E[\hat Y=1\mid A]$ should be $\mathbb P(\hat Y=1\mid A=a)$.
6. **ΔEO is non-standard.** Hardt's definition uses max. The mean of the FPR/TPR gaps is fine only if stated as a choice.

### 3.4 DP-GCCP (Eq. 15) — **CRITICAL**

1. **Wrong Laplace scale.**
   - The sensitivity of an empirical quantile of scores in $[0,1]$ is up to 1, not $1/|\mathcal S_a|$. Replacing one score can move the quantile to a neighbouring order statistic, potentially across the range.
   - Noise scale $1/(|\mathcal S_a|\epsilon)$ therefore under-noises by a factor of $|\mathcal S_a|$. The mechanism is *not* $\epsilon_{\text{conf}}$-DP.
   - Use the exponential mechanism (Smith 2011), a DP binary search or histogram CDF (Gupta et al. 2010), or the "Private Prediction Sets" approach (Angelopoulos, Bates, Zrnic, Jordan).
   - Line 79 of `src/conformal.py` uses `sensitivity/epsilon`; check what `sensitivity` is set to.
2. **Unit mismatch.**
   - $\tau_a$ is added in *probability-level* units: $\text{Quantile}_{1-\alpha+\tau_a}$.
   - The Laplace noise is in *score* units.
   - A level shift cannot compensate for value-space noise. Also, two-sided Laplace noise can be negative, so the "buffer" does not guarantee anything.
3. **The tail constant is unjustified.**
   - $2\ln 10$ gives a tail of $0.5\,e^{-2\ln10}=0.005$. This is neither motivated nor stated.
   - For $|\mathcal S_a|=30$ and $\epsilon_{\text{conf}}=1$, $\tau_a=0.15$ and the level is $1.05>1$, which is undefined.
4. **The standard finite-sample correction is missing.** Split conformal needs $\lceil(n+1)(1-\alpha)\rceil/n$ for the finite-sample guarantee.
5. **"Guarantee" is overstated.**
   - Group-conditional marginal coverage over calibration draws holds only if exchangeability holds *within each group*.
   - The paper's own result shows 88.3% for the minority group, below 90%. Yet the text says "guaranteeing valid coverage" and the abstract says "exact nominal 90%".
   - A noise-buffered quantile gives a probabilistic statement, not "exact".
6. **The method is Mondrian CP plus a private quantile.**
   - Mondrian conformal prediction (Vovk 2003; Vovk et al. 2005) and equalized coverage (Romano et al. 2019) already provide group-conditional validity.
   - Gibbs–Cherian–Candès 2023 and private prediction sets are missing from the citations.
   - The "non-trivial" novelty claim is overstated. Disjoint groups compose in parallel, so the composition is trivial.
7. **Privacy composition is not stated.**
   - Is $\mathcal D_{\text{cal}}$ disjoint from training? State total privacy as $(\varepsilon,\delta)$ for train data and $\epsilon_{\text{conf}}$ for calibration data.
   - $\epsilon_{\text{conf}}$ is never given.
   - "Without privacy leakage" in the abstract is false; DP bounds leakage but does not remove it.
8. **The group label is needed at test time.** Race or ethnicity may be unavailable or legally unusable at inference, and group-specific thresholds may be disparate treatment. This is not discussed.
9. **The score $1-\hat p_y$ ignores the epistemic evidence.** Conformalizing a function of $u$ would be more consistent with the paper's story.
10. **Calibration-set size.** With 270 patients, 3 folds and about 32% females, the female calibration set may have around 10–20 points. A 90% conformal quantile at that size has very coarse guarantees (granularity ≥ 1/(n+1)).

### 3.5 DCDA (Eqs. 16–17)

1. **Which class $k$ in Channel A?** It is unspecified (predicted or true class?).
2. **The channels are close to redundant.** For binary $K=2$, $\partial_j\ln\alpha_k=\partial_j e_k/\alpha_k$ and $\partial_j u=-K S^{-2}\sum_k\partial_j e_k$. Channel B is essentially the negative of the total-evidence gradient. The "decoupling" claim must be tested empirically (rank correlation of A and B).
3. **Missing features cannot light up.**
   - The baseline $\mathbf x^0$ is the "population mean vitals", and missing values are presumably mean-imputed.
   - Then $x_j-x_j^0=0$ and $\Phi_j=0$ exactly.
   - So Channel B **cannot** "identify missing serial troponin" as claimed in Section 6.5. Troponin is also not among the listed features.
   - BUN/creatinine *ratio* is also not a feature.
4. **IG completeness and the baseline choice** need discussion (completeness holds only for the chosen function and baseline). Give the number of integration steps.
5. **"Closed-form evidence gradients" is vacuous.** All gradients come from autograd, so nothing here is specific to Dirichlet models.
6. **Novelty.** Uncertainty attribution is established. Missing citations include Antorán et al. 2021 (CLUE), Perez et al. 2022 (attribution of predictive uncertainty), and Janzing et al. 2020. The statement that existing attribution frameworks are "single-channel" is incorrect.
7. **Conceptual error about DP and attributions.**
   - After training, a DP model has *fixed* weights. Attributions are not "stochastically perturbed" at explanation time.
   - Attribution degradation under DP comes from lower model quality or distorted decision boundaries, not noise at inference.
   - This claim is also tied to the cited prior work: Patel et al. 2020 and Saifullah et al. 2022 ("Privacy meets explainability") are missing.

### 3.6 Gap 1 SNR statement

$\text{SNR}_a=N_a\|\bar g_a\|/(\sigma C\sqrt d)$ is wrong on three counts:
- The noise is added to the *batch sum*, so the relevant count is the batch group count $n_a$, not $N_a$.
- It is linear in $N_a$, so "SNR$_0\to0$" does not follow.
- The noise norm is $\sigma C\sqrt d$ and is shared by all groups.

The better-supported mechanism (which the Related Work states) is that minority samples tend to have larger or misaligned gradients that are *clipped more*. That contradicts the Introduction's "noise floor" explanation.

### 3.7 Calibration claim

The paper says DP training causes "underconfidence". Bu et al. 2021 report that DP-SGD tends to cause *overconfidence* because of clipping. Support the direction with reliability diagrams and signed calibration error, and compare against post-hoc **temperature scaling** on a (private) validation set.

---

## 4. Clinical & Biomedical Accuracy

1. **The "Multi-Center Inpatient EHR Cohort" is synthetic — CRITICAL.**
   - `data/download_benchmarks.py` creates it from independent random draws, for example `np.random.normal(128, 22)` for systolic BP and `np.random.poisson(2.5)` for SOFA.
   - Features are independent, and the outcome is generated from a hand-written probability. There are no real temporal, correlational or multi-site structures.
   - The manuscript calls it "real-world", "scaled real-world", "Multi-Center" and "modeled after MIMIC-IV and eICU".
   - That is misleading. The subgroup disparity and the 29% minority are *design parameters* of the generator.
   - Every EHR result (Table 1 block 3, Table 2, the conformal section, Figs. 2–3, and the ICU claims) must be removed or relabelled "simulation". Replace it with real data (MIMIC-IV, eICU-CRD, which is genuinely multi-hospital, or other credentialed data).
2. **Sensitive attribute is undefined.**
   - "Underrepresented minority" is not a variable. Name it (race/ethnicity? insurance? language?).
   - Note that the repo's synthetic generator also draws sex independently (p=0.48) and uses a separate `sensitive` variable.
3. **Unrealistic or unsupported clinical details.**
   - D=15 but only 13 features are listed (5 vitals + 6 labs + 2 scores), plus age/sex in the generator.
   - SOFA is itself computed from creatinine, platelets, MAP/hypotension and other variables, so treating it as an independent feature leaks and duplicates information.
   - The outcome label (30-day mortality vs "acute clinical deterioration") is ambiguous; these differ greatly in prevalence and timing. Prevalence is not reported.
   - Lactate >4.0 mmol/L and SBP <85 mmHg are consistent with sepsis/shock criteria. "Refractory hypotension" is a specific clinical entity (persisting despite fluids/vasopressors) and is not measurable from a static SBP value.
   - "Lactic acidosis" needs pH/bicarbonate. Elevated lactate alone is hyperlactatemia.
   - A "borderline tachycardia with normal lactate" example has no supporting data.
4. **Statlog Heart.**
   - The original has 13 attributes, not 12. Say whether sex was removed.
   - "Real-world clinical cohort" understates that it is a 1980s Cleveland-derived dataset with 270 cases.
   - CAD prevalence differs strongly by sex. ΔDP then penalises clinically appropriate prediction differences.
5. **Pima Indians.**
   - The age split looks inverted. The median age is about 29, so ≤30 should be above 50%, not 46%. Please verify the 46/54 numbers.
   - Zeros in glucose, BP, skin-fold thickness, insulin and BMI are missing values. State the imputation.
   - Pima is a single ancestral population of women, where "age" as a protected attribute is not in the usual regulatory sense. Discuss the ethics of using Native American data.
6. **Clinical workflow.**
   - The $u<0.20$ threshold is arbitrary and untested.
   - Abstention rate, workload, and accuracy on the retained cases are not reported. There is no net-benefit or decision-curve analysis.
   - "Automated triage" with a mortality predictor is a high-risk use. It needs prospective and human-factors evaluation, and a discussion of automation bias.
   - Channel A and B outputs were never evaluated by clinicians.
7. **Regulatory statements are too strong.**
   - "Mathematically proven privacy bounds … guarantees compliance with HIPAA de-identification" is incorrect. DP is not equivalent to HIPAA Safe Harbor or Expert Determination.
   - GDPR Article 25 compliance is not established by a DP guarantee.
   - Fairness thresholds "ΔDP<0.10, ΔEO<0.10" are described as regulatory thresholds. No regulator defines them.
   - Section 1557 is an anti-discrimination rule and does not set the metric thresholds given. FDA guidance (PCCP, GMLP) and EU AI Act obligations should be cited precisely.
   - On the Introduction's pillars: HIPAA/GDPR "mandates … through formal guarantees" overclaims.

---

## 5. Statistical & Experimental Soundness

1. **Hyperparameters and protocol are missing.** Not stated: architecture, $K$, $d$, $M$, optimiser, learning rate, epochs, batch size and sampling, $C$, $\sigma$ for TRACER-Med, $\gamma$, $\lambda_{\text{fair}}$, $\lambda_{\text{reg}}$, $\epsilon_{\text{conf}}$, split ratios for train/cal/test, δ, and how ε=2.45 was obtained. Also unstated: whether all private models share the same ε. Only DP-SGD has σ=0.8. Without a matched ε, comparisons between DP-SGD, DP-EDL and TRACER-Med are meaningless.
2. **σ=0.8 is implausible for ε=2.45.** With small σ and many steps, ε is typically ≫10. Give $(q,T,\sigma,\delta)$ and the accountant (Opacus RDP/PRV?).
3. **Statistical power and std values.**
   - With 3-fold CV on 270 patients, each test fold has ~90 patients and ~29 females. A subgroup coverage gap or ΔDP is quantised and has an SE of several percentage points.
   - Reported stds of ±0.01–0.02 (and "±0.00" for infidelity) are not credible for subgroup metrics.
   - State whether the std is across seeds, folds, or both.
   - Provide bootstrap CIs, paired tests, and effect sizes.
   - Differences like 0.509 vs 0.512 MIA AUC (bolded) are noise.
4. **Missing and weak baselines.**
   - DP-SGD + SWD (private fair-ERM).
   - Existing fair-DP methods: Tran et al., Xu et al., Esipova et al., Jagielski et al. 2019.
   - DP-SGD + temperature scaling, deep ensembles, MC-dropout, and conventional Mondrian CP.
   - Strong tabular DP baselines: DP logistic regression, DP-GBDT.
   - Group-DRO / reweighting under DP.
   - Non-private Mondrian CP as an upper bound.
5. **Utility numbers look too good.**
   - TRACER-Med (DP, ε=2.45) gets 0.865 AUC against DP-EDL's 0.824, using only a clipping reweighting plus a *utility-costing* fairness regulariser.
   - It loses just 0.019 to non-private ERM, on only 270–768 samples (Statlog: ΔAUC 0.021).
   - On Statlog, DP-SGD at σ=0.8 loses ~8 AUC points but TRACER-Med loses 2. Explain the mechanism, or show the learning curves and ε-sweep (ε ∈ {0.5, 1, 2, 4, 8, ∞}).
6. **Inconsistent counts.** The text says "eight competitive architectures". The list is five baselines + proposed + three ablations = nine rows, and the main table shows six.
7. **The headline "74%" is not reproducible.**
   - The Abstract and Contributions say "up to 74%". Section 6.2 says 78%.
   - "DP-induced disparity" is the DP-SGD minus ERM gap, which is 0.078 on EHR. TRACER-Med's ΔDP (0.074) is *below* non-private ERM (0.264), so you "eliminate" over 100% of it.
   - ΔEO gives 72.5%, 73.7% and 75.6% against DP-SGD, so 74% is plausibly an average of those, which is a different quantity. Define it precisely, and use one definition consistently.
8. **Coverage comparison is apples-to-oranges.** "Reducing coverage disparity from 21.4% to 3.1%" compares a DP-SGD model + marginal CP against TRACER-Med + group CP. Separate the model effect from the conformal effect. The ablation (16.7%) is the correct comparison.
9. **Missing conformal metrics.** Per-group set size, singleton, empty-set and conditional-on-label coverage, plus coverage by *class* (class-conditional coverage matters for rare disease). Coverage vs ε_conf curve. Figure 2 shows no confidence intervals. Show the coverage distribution over repeated splits.
10. **Privacy audit is weak.**
    - MIA AUC near 0.5 does not corroborate an ε bound. Shokri/Yeom loss-threshold attacks are weak.
    - Use LiRA (Carlini et al. 2022) and report TPR at low FPR (e.g. 0.1%), and canary-based auditing (Jagielski et al. 2020; Nasr et al. 2021).
    - Report per-group MIA. Fairness and privacy can trade off (Chang & Shokri 2021).
    - ERM's MIA of 0.62 on N=10,000 is high for tabular data of that size, which again hints at the generator.
11. **Explanation evaluation.**
    - "Faithfulness Rank Correlation" appears in the metric list but never in any table.
    - Infidelity is Yeh et al. 2019 and is mis-attributed to Sundararajan et al. Infidelity depends on the perturbation distribution, which is unspecified.
    - There is no comparison with IG on logits, SHAP, or saliency on the same models.
    - No sanity checks (Adebayo et al. 2018) and no deletion/insertion tests.
    - No evaluation of Channel B at all, and no clinician study.
12. **Ablation findings.**
    - Ablations are single-component removals only. Add pairwise ablations and a γ-sweep, plus SC-DP's separate effect on ECE.
    - w/o SC-DP (0.218) is *better* than DP-EDL (0.315) yet uses the same global clipping. That difference is due to SWD, confirming that SC-DP is not the main driver.
    - "Proving that … is indispensable" is too strong for an empirical result.
13. **Reproducibility.** Release code, seeds, accountant configuration, and the data-generation script with an honest label. The "RTX 5000 Ada (16GB)" description should be checked (Ada generation RTX 5000 has 32GB; the 16GB card is an RTX A4000/other). Intel "Core i9 platform" is not informative.
14. **Overclaiming language.** Remove or soften "first unified architecture", "conquer", "completely inverts", "slashes", "Exhaustive", "mathematically proven", "guarantees". The Trilemma is a framing, not something "proved" ("proving why isolated interventions fail" in Contribution 1 has no theorem).
15. **The "Trilemma" is actually four items.** It is bulleted as four effects but called a trilemma of privacy, fairness and uncertainty. The "Conformal" and "Attribution" bullets are separate. Align the naming and the content.

---

## 6. Minor Issues, Writing, LaTeX Syntax, Micro-Edits

### LaTeX / formatting
1. `\documentclass[preprint,12pt,3p]{elsarticle}`: `preprint` with `3p` is contradictory (`3p` is a review/final layout option). The `\affiliation` command with `organization=` requires elsarticle ≥ 3.0, so confirm the class version. Compile with pdflatex as you do.
2. `\usepackage{algorithm}` + `\usepackage{algorithmic}`: Elsevier templates generally prefer `algorithm`/`algpseudocode`. `\REQUIRE/\ENSURE` work with `algorithmic`, but check the `\STATE ... \FOR` nesting.
3. `\usepackage{amsthm}` is loaded, but Theorem 1 is typed as `\noindent \textbf{Theorem 1 ...}` rather than a `theorem` environment. Define `\newtheorem` and use it for numbering and cross-references.
4. `hyperref` should be loaded last, and `cleveref` is not needed. `\usepackage{url}` is redundant with hyperref.
5. Missing `\bibliography{references}` entries were not verifiable (no `.bib` supplied). Check every key: `kompa2021second`, `farrand2020neither`, `xu2023fair`, `feldman2021individual`, `tibshirani2019conformal`, `dong2022gaussian`, `charpentier2020posterior`.
6. `\begin{frontmatter}`: there is a `\cortext` but no `\corref` placement error; confirm the corresponding-author mark renders. Add `\ead`s for the other authors, if any. The affiliation "Technology Science Complex" is not a standard address; add a postal code.
7. `\resizebox{\textwidth}{!}{...}` on a 9-column table shrinks the text. Use `\small`/`\scriptsize` and `tabularx` or split the table. Rescaling inside `table*` with 12pt body may make it unreadable.
8. Table 1: the bold in the "MIA AUC" column marks DP-SGD only. The caption says "Best values among private models are bolded", but the non-private rows are not excluded from visual comparison. Add a "Private?" column or horizontal rule. The std "±0.00" should be "<0.005".
9. Table 1 / Table 2 column headers use `$\Delta_{\text{DP}}$` (see notation clash). In Table 2 the header says "Cov. Disparity" but Table 1 says "Cov. Gap". Use one term.
10. Equation numbering: some displayed equations are unnumbered. In the SC-DP section, the unlabeled equations should be labelled for reference. Use `\label`/`\eqref`.
11. In Section 4.4 and Algorithm 1, the DP-GCCP formula is duplicated verbatim. State it once and reference it.
12. Figure 1 is `fig4_trustworthiness_profile.pdf` and is used as the "schematic overview" referenced as the architecture diagram. It is a radar chart of results and is not a schematic. A true architecture diagram is needed. Figure numbering/file names (fig4 used as Fig. 1) invite confusion.
13. Figures use `figure*` with `[t]` while others use `figure`; mixed widths (0.95, 1.0, 0.88). Standardise. Give all figures alt text.
14. Missing `\usepackage{siunitx}` for units (mmol/L, mmHg). Use `\,` spacing consistently ("$>4.0$ mmol/L").
15. The abstract is too long (≈220 words and many numeric claims). Elsevier AIiM prefers a structured abstract (~250 words max) and highlights (3–5 bullets, ≤85 chars each) are missing. A graphical abstract and the required declarations (CRediT, declaration of competing interests, data availability, ethical approval, generative-AI use) are missing.

### Notation and consistency
16. Overloaded symbols: $\alpha$ (Dirichlet / miscoverage / RDP order), $\theta$ (weights / projections), $\mathcal D$ (data / KL), DP (parity / privacy), $\sigma$ (noise / not-defined elsewhere), $a$ (group / attribute), $S$ (strength) vs $\mathcal S_a$ (score set), $C$ (clip) vs $\mathcal C(X)$ (set) vs $C_a$.
17. $D$ is used for feature dimension and $d$ for latent dimension, but $\mathcal D$ for data and $\mathcal D'$ for neighbours. $B$ appears in $\sum_{i=1}^B$ (Eq. 6) while $|\mathcal B|$ is used elsewhere. Use one.
18. Eq. 6's per-sample loss uses $y_{ik}$ with $\psi(S_i)-\psi(\alpha_{ik})$; this is correct, but the same $\mathcal L_{\text{EDL}}$ is called with different arguments in Algorithm 1 ($\mathcal L_{\text{EDL}}(\mathbf e_i,\boldsymbol\alpha_i,y_i)$).
19. $u\in(0,1]$ is correct, but "$S\gg K$ yields $u\to 0$" and "$\mathbf e\to\mathbf 0$ yields $u\to1$" is fine. State that for $K=2$ vacuity is bounded below by evidence scaling.
20. Algorithm 1 line "Initialize … with LayerNorm normalizations": LayerNorm is the DP-compatible choice (BatchNorm is not), which should be explained. Also the architecture, `softplus` output and $f_\theta$ vs $h_\theta$ are used inconsistently ($f_\theta$ = evidence network; $h_\theta$ = penultimate).
21. In the Algorithm, `Forward pass: e_i, α_i, p_i, u_i = f_θ(x_i)` uses $\mathbf p_i$, while elsewhere $\hat p$ is used. `Account cumulative privacy` is outside the loops, while the accountant should be updated per step.
22. The algorithm splits into train/cal/test *within* the DP pipeline but does not say which privacy budget covers the split.
23. Section 3.2 Gap 1: "$N_0 \ll N_1$" but the cohorts are 29/71 or 32/68 (not "≪").
24. "Clinical Equity Zone (green shaded area)" in Fig. 1 caption is undefined in the text.
25. Section 4.1 bullets: "12 physiological features (… fluoroscopy vessels)" — fluoroscopy vessels is a count of vessels coloured, not "physiological".

### Writing / phrasing
26. Over-promotional adjectives throughout: "wildly degrades", "catastrophic drop", "destructive interference", "disappointingly sparse", "triumphs", "conquer", "alarming", "negligible", "completely inverts", "slashes". Replace with measured, quantitative language.
27. "Four cardinal pillars" are listed, but the "Trilemma" is introduced as three. Reword.
28. "DP-SGD acts as an equity penalty" → "has disparate impact".
29. "Standard softmax neural networks trained under DP suffer from severe probability distortion and underconfidence" → cite or soften (see 3.7).
30. "Gap 1 (The Disparity Amplification Paradox)": it is not a paradox. "Gap 4 (Single-Channel Attribution Breakdown)" is not demonstrated.
31. "Provide provable privacy" vs "satisfying legal mandates": legal compliance should not be asserted.
32. Related Work: "clinical implementations that bridge DP with second-order uncertainty quantification remain unaddressed" is a strong claim; add citations or soften (DP + Bayesian/UQ work exists).
33. "Fair representation learning via optimal transport" cites Zemel 2013 (not OT) and Kolouri 2019 (generalized sliced Wasserstein, not a fairness paper). Replace with, e.g., Jiang et al. 2020 (Wasserstein fair classification), Chiappa et al. 2021, or similar.
34. "Mamba" in Future Work is a speculative buzzword with no citation.
35. "TRACER" acronym: "Trustworthy, Accredited, Conformalized, Evidential, and Differentially pRivate" — "Accredited" is not a technical property. Also the second "D" and "L" ("Deep Learning") are not captured, and the typesetting `P\textbf{R}ivate` is awkward.
36. Section 6.4: "corroborating the theoretical Rényi DP bound … with empirical privacy defense" → empirical attacks cannot corroborate a theoretical bound.
37. Section 6.3: "directly violating clinical safety thresholds" has no clinical safety threshold defined.
38. "Pareto-optimal frontier" (Fig. 1 caption): no Pareto analysis is shown, and the claim "consistently anchors the Pareto frontier" is false where non-private models dominate on AUC.
39. "10,000-patient … ($N=10{,}000$)" — use consistent `10{,}000` or "10,000" formatting; "$\text{SpO}_2$" is fine.
40. Capitalisation and hyphenation inconsistencies: "R\'enyi" is spelled correctly, but "Differentially P\textbf{R}ivate" and "D-EDL" (used once, never defined) should be fixed. "Standard DP-SGD/EDL" use "Standard" inconsistently with the model names ("ERM" vs "Fair-ERM").
41. The abstract's "ROC-AUC $0.85\pm0.02$" is not any single value in Table 1 (0.841, 0.812, 0.865). It must say it's an average, or name the dataset.
42. Missing space/punctuation: "`... \cite{abadi2016deep,mironov2017renyi}.`" fine; several sentences end with a citation before the period inconsistently. In 4.1: "(…, 32\% minority; $a=1$: male, 68\%)" — put the percentage in the same format for both groups.
43. "Related Work and Theoretical Landscape": the "theoretical" part contains no theory. Rename.
44. The first-person "we prove" in the Contributions: there is one (flawed) theorem and no proof for coverage, fairness, or the 74% claim. DP-GCCP "mathematical finite-sample coverage guarantees" is asserted with no theorem.

---

## 7. Final Verdict & Checklist of Actionable Revisions

**Verdict: Major revision (not acceptable in current form).** The framing is relevant. However, a core theoretical claim is wrong, the conformal mechanism is not DP and not guaranteed, and the principal benchmark is synthetic but presented as real EHR data.

### Must fix (blocking)
- [ ] **Rewrite or retract Theorem 1.**
  - Make clip bounds public-weight and bounded: $\max_a\tilde w_a\le1$, or scale noise by $\sigma\max_a C_a$.
  - Remove the batch-count dependence, or account for it.
  - Use a correct subsampled-Gaussian RDP analysis (Mironov et al. 2019), include the RDP→(ε,δ) conversion, and report $(q,T,\sigma,\delta,\varepsilon)$ for every private model at matched ε.
- [ ] **Fix the SWD/DP interaction.** Define $\mathcal L_{\text{fair}}$ as a batch-level term; show it has bounded sensitivity, or apply it only to public/non-sensitive proxies, or drop the DP claim for that term.
- [ ] **Replace the Laplace-on-quantile mechanism** with a valid DP quantile (exponential mechanism or DP binary search/histogram). Use the $\lceil(n+1)(1-\alpha)\rceil/n$ correction. Fix the unit mismatch and cap levels at 1. Cite Mondrian CP, Angelopoulos et al. (private prediction sets), Romano et al., Gibbs et al. State $\epsilon_{\text{conf}}$ and total budget composition.
- [ ] **Stop calling the EHR cohort real or multi-center.** Either (a) label it explicitly as a simulation throughout (title, abstract, Section 5, Fig. captions) and drop the clinical claims drawn from it, or (b) rerun everything on MIMIC-IV and/or eICU-CRD with a named real protected attribute.
- [ ] **Correct the headline claims.** Define "74% of DP-induced disparity" precisely; reconcile it with the 78% in Section 6.2. Remove "exact nominal 90%", "guarantees", "without privacy leakage", and "first … to conquer".
- [ ] **Add the missing controls:** DP-SGD + SWD, existing fair-DP-SGD methods, DP + temperature scaling, Mondrian CP non-private and on the non-DP model. Run an ε-sweep.

### Should fix (high)
- [ ] Provide full hyperparameters, splits, accountant and code. Report CIs/bootstraps, and use paired tests; acknowledge small-subgroup noise (Statlog/Pima cal-set sizes).
- [ ] Strengthen the privacy audit with LiRA / TPR@low-FPR and canary auditing, plus per-group MIA. Do not equate chance-level MIA with a proven ε.
- [ ] Define aleatoric uncertainty; discuss the EDL pitfalls literature; anneal and specify $\lambda_{\text{reg}}$.
- [ ] Fix DCDA: specify $k$; show that A and B are not redundant (rank correlation); handle missingness with explicit missing indicators or masked baselines; add baselines (IG on logits, SHAP), sanity checks, deletion tests, and clinician evaluation. Cite uncertainty-attribution literature. Remove the "DP noise perturbs explanations at inference" argument or support it empirically.
- [ ] Re-justify the DP-vs-fairness narrative: correct the Gap 1 SNR formula; reconcile "noise floor" versus "clipping bias" explanations.
- [ ] Reconcile demographic-parity alignment with equalized-odds reporting when base rates differ (Statlog).
- [ ] Verify the Pima age-split proportions, the Statlog feature count, the D=15 feature list, SOFA leakage, outcome definition and prevalence.

### Should fix (medium)
- [ ] Tone down regulatory statements (HIPAA/GDPR/EU AI Act/§1557); remove invented "regulatory thresholds".
- [ ] Discuss use of sensitive attributes at inference, disparate treatment from group-specific thresholds, and calibration of $u$ thresholds. Report abstention rate and workflow metrics.
- [ ] Repair citations: infidelity (Yeh et al. 2019), OT-based fair representations, DP/miscalibration direction (Bu et al.), SWD (Bonneel; Kolouri 2018/2019).
- [ ] Resolve "eight baselines", the Fig. 1 schematic/radar mismatch, and Table 1 formatting (bold semantics, ±0.00, `\resizebox`).
- [ ] Disambiguate notation (α, θ, 𝒟, DP, $B$ vs $|\mathcal B|$) and use a real `theorem` environment.

### Polish (low)
- [ ] Add Highlights, structured abstract (≤250 words), CRediT, data/code availability, ethics and competing-interest statements, and generative-AI disclosure per Elsevier policy.
- [ ] Replace promotional language with measured wording; fix hardware description; unify "Cov. Gap/Disparity"; define "Clinical Equity Zone", "D-EDL", and fix the acronym typesetting.
- [ ] Rename "Related Work and Theoretical Landscape" and add a threats-to-validity paragraph (synthetic data, small subgroups, single-site assumptions, exchangeability).

If helpful, I can apply the fixes (corrected Theorem 1 and RDP accounting, a valid DP quantile in `src/conformal.py`, a DP-SGD+SWD baseline, and relabelling the synthetic cohort) to the manuscript and code. Say which you want first.
