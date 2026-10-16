# Peer Review: TRACER-Med

Review of the LaTeX source as supplied. I could not see the figures, the `.bib` file, or the code. Findings about those are flagged as "verify".

The source title is "TRACER-Med: Trustworthy Deep Learning for Differentially Private, Equitable, and Uncertainty-Calibrated Clinical Decision Support". The title in your request is longer ("A Unified Trustworthy Deep Learning Framework… with Dual-Channel Explainability"). Make sure the submission system and the source use the same title.

---

## 1. Overall Assessment & Summary of Contributions

The manuscript proposes TRACER-Med, which combines five pieces:
- An evidential Dirichlet network (EDL).
- Group-adaptive clipping in DP-SGD (EF-SC-DP).
- A class-conditional sliced-Wasserstein latent-alignment penalty (CC-SWD).
- A private histogram-based group-conditional conformal calibration (DP-GCCP).
- A two-channel Integrated-Gradients attribution (DCDA).

It is evaluated on two small UCI/OpenML tables (Heart, N=270; Pima, N=768) and one simulated EHR cohort. The motivating "Privacy–Fairness–Uncertainty Trilemma" is a sensible framing for clinical AI.

**The manuscript is not publishable in its current form.** There are four main reasons:
1. **Both privacy proofs have gaps.** The Theorem 1 sensitivity argument fails once the batch-dependent CC-SWD gradient is included. The Theorem 2 privacy and PAC arguments each have a defect (§3).
2. **Headline numbers are inconsistent.** The abstract, governance section and conclusion cite results that do not match Table 1 (§3.8).
3. **The data contradict the central claims.** Table 1 and the ablation show TRACER-Med is not best on most metrics, and its Equalized Odds is worse than DP-EDL on Heart. DP-SGD also *reduces* disparity relative to ERM, so the motivating premise is not reproduced (§5).
4. **The evidence is statistically and clinically thin.** It rests on 3 folds, tiny minority groups, an unrealistic simulated cohort, and no comparison with the fair-DP methods the paper cites (§4–5).

---

## 2. Major Strengths

- The trilemma framing, the focus on Equalized Odds under unequal base rates, and the separation of aleatoric from epistemic uncertainty fit clinical needs.
- The Dirichlet quantities are mostly correct. The aleatoric entropy, mutual information and Bayes loss are right, and so is the KL closed form. The logit-gradient derivation (§4.2) is also correct: ∂L/∂z_j = (ψ₁(S) − y_jψ₁(α_j))·σ(z_j).
- The authors use add/remove adjacency and Poisson subsampling. They use LayerNorm rather than BatchNorm, which keeps per-sample gradients well defined.
- The histogram mechanism (Δ₁=1 under add/remove, Laplace noise, post-processing) is the right basic design for private conformal calibration.
- The paper is open about the DP-GCCP design and is explicit that the MIA audit gives only a lower bound.
- The regulatory discussion is unusually thorough for a methods paper. The ethics, code-availability and CRediT sections are present.

---

## 3. Critical Methodological & Mathematical Issues

### 3.1 Theorem 1 does not cover the CC-SWD gradient (CRITICAL)
- The adjoint vector uᵢ = ∂L_CC-SWD(B)/∂hᵢ depends on **all records in the batch**. The sorted order, the quantile matching and the (a,y) cell membership all do.
- Adding or removing one record r* therefore changes the clipped gradients g̃ⱼ of other records, through their uⱼ. The proof's claim that the summed-gradient difference is "either **0** or ±g̃*" is false.
- The actual sensitivity can be up to ≈ |B|·C_max in the worst case, so σC_max noise does not give the stated guarantee.
- The claims "without auxiliary privacy leakage or batch contamination" and "exact individual gradient contribution" (§4.3) are unsupported. The chain-rule decomposition is algebraically true but irrelevant to DP, which needs gᵢ to be a function of record i alone.
- **Fixes:**
  - Compute the alignment term from a separate privatized statistic. For example, clip and noise per-(a,y) latent moments or sketches.
  - Or bound the cross-record influence explicitly and recalibrate the noise.
  - Or drop the "private" claim for CC-SWD.
- Empty or near-empty (a,y) cells are likely. With batch 32 and 32% minority, the (minority, positive) cell has about 4–5 samples, and Poisson sampling can leave it empty. The paper does not say how this is handled.

### 3.2 Group-adaptive clipping is dominated by uniform clipping at C_max (HIGH)
- Noise is scaled to C_max = max_a C*_a, so no group's privacy cost falls when its own threshold is lowered.
- Compared with uniform clipping at C = C_max, EF-SC-DP gives every group a threshold ≤ C_max at identical noise. It only discards signal.
- The valid benchmark is "DP-SGD with uniform C tuned, including C = C_max". Compared with C₀=1, any gain may just be a larger effective C or implicit reweighting. It is not "preserving minority signal".
- The weights contain no Fisher-metric term. They use only ratios of mean evidence and prevalence, so "Evidential-Fisher" and "Dirichlet curvature" are labels, not derivations. Fisher information is a metric; curvature is something else.
- S̄_a ("prior expected evidence") and p_a are never given numerically, and their source is unclear. If they come from the training data, there is unaccounted privacy leakage. The Pima/NHANES prior also does not match the Pima population.
- The mechanism ("clipping suppresses minority evidence; ψ₁ makes low-evidence gradients large") is asserted, not measured. Report per-group gradient norms, clip fractions and mean S.

### 3.3 Theorem 2 (DP-GCCP): privacy proof (HIGH)
- τ_a and the quantile q̂_a depend on the true count n_a. Under add/remove adjacency n_a changes, so releasing q̂_a is **not** pure post-processing of the noisy histogram. The (ε_conf, 0)-DP claim fails as written. Privatize n_a at extra cost, or restate the adjacency as per-group substitution.
- The shrinkage rule q̂ᵃ_reg = λ_a q̂_a + (1−λ_a) q̂_marg (§4.4) has two problems:
  - It needs a marginal threshold q̂_marg computed from the same calibration data. That is a second ε_conf release, so total privacy cost is not ε_conf.
  - It destroys the PAC guarantee, because a convex combination of thresholds carries no coverage guarantee.
- State the overall guarantee for model plus calibration. Parallel composition over disjoint D_tr and D_cal gives max(ε, ε_conf), not a sum, if the split is data-independent. The text ("disjoint calibration budget") is ambiguous.
- Hyperparameter search, early stopping and S̄_a all consume budget that is not accounted for.

### 3.4 Theorem 2: PAC proof (HIGH)
- The Laplace bound controls max_m |ξ_m|/n_a for a **single bin**. The CDF error is a **cumulative sum** of up to M_bins noises, so the error is roughly M·max|ξ|/n_a, not max|ξ|/n_a. The second inequality in the final chain is unjustified.
- Two other noise effects are ignored:
  - Normalizing by the noisy total max(1, Σh̃).
  - The positive bias from max(0,·). Each empty bin gains about 1/(2ε) expected mass, so M=20 bins adds roughly 10 spurious counts.
- The PAC statement is only valid conditional on a fixed trained model. Say so.

### 3.5 The conformal buffer makes sets vacuous at the dataset sizes used (HIGH)
- Take α=0.1, β=0.05 and an assumed M_bins=20 (the paper never states M_bins).
- At n_a=20, τ_a ≈ 0.33 + 0.33 = 0.66. At n_a=100, τ_a ≈ 0.15 + 0.07 ≈ 0.22.
- The target 1−α+τ_a exceeds 1 until n_a ≳ 350 per group. Then inf ∅ = 1 and q̂_a = 1, so the prediction set is always {0,1}.
- The calibration split sizes are never reported. For Heart with 3-fold CV and a minority of 32%, the calibration minority group is likely about 10–20 records.
- So the theorem as written yields trivial sets on Heart and Pima. This probably explains the "0.000 ± 0.000" gaps in Table 1, and the shrinkage patch for n_a<25 can only be an implementation deviation. Either the reported results do not use the theorem's rule, or they use trivial sets.
- **Average set cardinality** and **marginal coverage** are listed as metrics but never reported. They are needed.
- A binary task also needs a note on empty sets: if q̂<0.5 and p̂≈0.5, C(x)=∅.
- Coverage is conditional on group, not on class. For screening, coverage on positives is what matters, so Mondrian (a,y) calibration is the clinically relevant version. Vovk's Mondrian/class-conditional CP is not cited.
- The group a must be known at test time. That raises practicality and legal issues about using protected attributes in deployed decisions.

### 3.6 CC-SWD formulation (MEDIUM)
- The symbol 𝒲₁ is used for the 1-Wasserstein metric and then for the sliced version. Use SW₁. μ_{a,y} is undefined.
- Finite-M_proj SW is a Monte-Carlo estimate, so "exact analytical gradient of the joint objective" overstates. The quantile integral with unequal cell sizes needs an interpolation rule. M_proj and λ_fair are never given.
- "Transport" is a misnomer: no map is computed, so this is a distribution-matching penalty. Contribution 3 says "surrogate gradient" while §4.3 says "exact", which is inconsistent.
- The sum over y is unweighted, so rare cells count as much as common ones.

### 3.7 EDL and DCDA (MEDIUM)
- Bengs et al. (2022) show that EDL epistemic estimates are not reliably epistemic. The paper cites this and then moves on. Under DP, evidence suppression systematically inflates vacuity. u<0.20 (§7.2) is therefore not a validated epistemic signal.
- IG completeness holds only up to quadrature error (M_quad=50), and attributions are path- and baseline-dependent. "Exact probability delta driven by each biomarker" overclaims.
- The claim that median baselines "prevent off-manifold interpolation across categorical variables" is wrong. Medians of ordinal-coded chest-pain type or thal still give meaningless straight-line interpolants. Categoricals appear to be ordinal-coded (D=12 implies no one-hot). Use one-hot or a categorical-aware baseline.
- Medians computed from the data are an unaccounted release. A sex-matched baseline also uses the protected attribute.
- Channel B has no validation. Test it with feature-masking, OOD or synthetic-uncertainty experiments. Infidelity is scale-dependent and not comparable across models with different output variance. The promised "Faithfulness Rank Correlation" is never reported.

### 3.8 Numerical inconsistencies in the manuscript (CRITICAL)

| Quantity | Abstract / elsewhere | Table 1 |
|---|---|---|
| Heart ΔEO | **0.083 ± 0.02** (abstract, §7.1) | 0.158 ± 0.04 |
| Heart ΔSP | 0.232 ± 0.04 | 0.242 ± 0.02 |
| Heart ROC-AUC | 0.863 ± 0.05 | 0.875 ± 0.02 |
| Pima AUC / Brier | 0.826 / 0.173 | 0.817 / 0.180 |
| Heart coverage gap | 0.045 ± 0.05 | 0.024 ± 0.020 |
| EHR coverage gap | 0.009 ± 0.004 | 0.028 ± 0.021 |

- "42% EO reduction vs DP-EDL" (abstract, conclusion) is impossible. DP-EDL's ΔEO is 0.115, lower than TRACER-Med's 0.158.
- "27% SP reduction" does not match the 18.2% in §6.2.
- The abstract numbers look stale from an earlier run. Rebuild them from the final results file.

### 3.9 Smaller technical points
- Theorem 1 asserts RDP at integer orders. Newer RDP→(ε,δ) conversions are tighter. Name the accountant, such as Opacus or PRV, and its version.
- Describe how per-sample gradients with batch-level SWD are implemented. Report overhead.
- Algorithm 1 omits DCDA and the S̄_a/p_a estimation. The fifth module is absent from the "unified pipeline".
- The Algorithm and Theorem 1 present Poisson batches within "epochs", which is a common loose description. State steps T explicitly.

---

## 4. Clinical & Biomedical Accuracy

### Statlog Heart
- The cohort is an angiography-referral population with verification and selection bias. It is not a screening population, which conflicts with the "early disease screening" motivation.
- "Collected at Cleveland Clinic, donated to UCI in 1988" describes the Cleveland dataset (303 records). Statlog's 270-record variant came from a different donation (verify).
- Sex is a strong clinical risk factor. In this table, men have roughly 52% prevalence vs about 29% for women (approximate; verify). The oracle SP therefore equals the base-rate gap, about 0.24.
- Heart ΔSP of 0.242 is essentially at that level. That is not evidence of improved equity. Report base rates and oracle SP, and do not present SP as the headline result.

### Pima
- All participants are women aged ≥21 from one tribal population. "Age ≤30 vs >30" is a clinical risk factor and an arbitrary split. Aligning conditional latents across age may remove legitimate pathophysiology.
- The dataset has physiologically impossible zeros in glucose, BP, skin-fold, insulin and BMI. The handling is not described.

### Simulated EHR (HIGH)
- **Prevalence ≈ 42%** is unrealistic for ward deterioration (~2–15%) or ICU mortality (~10–20%). It makes the task unusually easy, and Brier and PR-AUC are uninterpretable.
- **Predictors:**
  - Arterial pH and PaO₂/FiO₂ are rarely drawn on wards.
  - Total bilirubin is rarely measured on wards.
  - SOFA is a composite of several listed variables (MAP, creatinine, bilirubin, platelets, P/F), so including it is redundant and collinear.
  - SOFA is ordinal, not "continuous".
  - Age, GCS, urine output, vasopressor status, comorbidity and informative missingness are absent.
- The outcome "ICU transfer or 30-day mortality" needs an index time, a horizon, and an exclusion for patients already in the ICU.
- The protected attribute is unspecified: race, insurance, sex? The "access disparity" mechanism is also unspecified (measurement bias or label bias?).
  - If the labels are biased by construction, Equalized Odds is not a valid clinical target. Equalized Odds assumes valid labels, the same caveat as in Obermeyer et al.
- The generator's correlation structure is not described. Channel A "finds" lactate, MAP and SOFA, which are exactly what the simulator encodes, so DCDA's clinical validity is not demonstrated.
- §6.1 calls the simulated cohort "multi-center Clinical EHR". It is neither multi-center nor clinical data.
- Real alternatives: actual MIMIC-IV/eICU (credentialed), PhysioNet Challenge 2019 sepsis, UCI Diabetes-130, NHANES/BRFSS. Check the data-use terms on any derived distribution.

### Thresholds and workflow
- Lactate >4 mmol/L, MAP <65 and SBP <90 are reasonable clinical thresholds.
- The u<0.20 rule means S>10, an arbitrary and untested cutoff. No abstention rate, no singleton rate and no selective-risk analysis is reported. With AUC ≈ 0.75, most sets at 90% coverage will be {0,1}, meaning near-total abstention.
- Abstention rates could differ by group, shifting review burden onto minorities. This needs an equity analysis.
- No clinician evaluation exists, so "co-pilot" is speculative.

### Regulatory section
- HIPAA and GDPR do not "require mathematical privacy guardrails" (Intro, item 1). DP is also not an anonymization guarantee at ε=15.
- **ε = 15.36** (Heart) and 6.13 (Pima) are weak guarantees. e^15 is about 4.7 million. Pair the headline with a target ε≤1–8 setting or a privacy–utility curve.
- The GMLP mapping overreaches:
  - Principle 3 is about representative data, not an algorithmic mitigation.
  - Principle 10 is about monitoring and retraining risk, not DP.
  - DP across continual retraining composes, which is not discussed.
- The PCCP guidance is final, "Marketing Submission Recommendations for a PCCP for AI-Enabled Device Software Functions" (Dec 2024). The "AI-Enabled Device Software Functions" lifecycle guidance is a separate draft (Jan 2025). The text conflates the two titles (verify).
- The EU AI Act high-risk obligations for Annex I devices have staggered dates, and amendments have been proposed. HHS §92.210 has had a 2025 compliance date and a changing legal environment. State the status as of today (2026-10-15) and use "supports documentation of…" rather than "supports compliance".
- Suggested additional frameworks: TRIPOD+AI (2024) and FUTURE-AI.

---

## 5. Statistical & Experimental Soundness

1. **Motivating premise not reproduced.** DP-SGD *lowers* ΔSP and ΔEO vs ERM on all three datasets (Heart EO 0.214→0.135, Pima 0.302→0.233). §6.2 says DP-SGD and DP-EDL "exhibit elevated fairness disparities". The table says the opposite. The section title "Mitigation of the Privacy–Fairness Trade-off" has nothing to mitigate in this data.
2. **TRACER-Med is not the best model.**
   - On Heart, DP-EDL wins 6 of 9 columns. It beats TRACER-Med on AUC, ECE, ΔEO, MIA, coverage gap and Brier. TRACER-Med wins only SP, and infidelity is tied.
   - On EHR, TRACER-Med is worst among private models on SP (0.021 vs 0.010) and coverage gap (0.028), and ties or loses on everything else.
   - On Pima, ΔEO 0.191±0.01 vs 0.235±0.08 is within noise. SP/MIA/ECE differences are ≤0.01.
3. **Insufficient replication.** "3-fold CV with seeds 42–44" is ambiguous (3 runs or 9?). The Heart test fold has 90 patients, about 29 female and about 8–13 female positives. TPR SE is about 0.1–0.15, yet stds of 0.01–0.04 are reported. Use repeated stratified CV (≥10×5), bootstrap CIs, paired tests, and effect sizes.
4. **Ablation (Table 3).**
   - "w/o DP-GCCP" cannot change a model's AUC, ECE, SP or EO, since conformal calibration is post hoc. Its rows differ from "Full" (0.881/0.100/0.261/0.118 vs 0.875/0.095/0.242/0.158). That shows run-to-run variance exceeds the claimed ablation effects.
   - Two rows share an identical coverage disparity (0.008 ± 0.011).
   - The "w/o SC-DP" ΔSP of 0.318±0.03 equals standard DP-SGD's ΔSP exactly, though the model differs. Check for result reuse.
   - Removing SWD or SC-DP *improves* AUC, ECE, ΔEO and coverage gap, and only SP worsens. The text reports only the SP side. Removing GCCP *improves* the coverage gap (0.008 vs 0.024), contradicting the equity claim.
   - The claim "subgroup-calibrated clipping is essential for coverage equity" rests on 0.037±0.052 vs 0.024±0.020, which is not significant.
   - The ablation covers Heart only, though the text says "across the benchmarks".
5. **Calibration claim contradicted.**
   - On EHR, ECE is 0.145 for TRACER-Med, vs 0.043 for ERM. The text's "well-calibrated Brier scores (0.224–0.226)" is false.
   - The constant-prevalence Brier at 42% is 0.2436, so these models are barely better than trivial, and DP-SGD (0.254) is worse.
   - TRACER-Med's ECE matches DP-EDL's (0.145 vs 0.146), so EF-SC-DP contributes nothing to Trade-off 2, and the ablation's w/o SC-DP is better on ECE (0.081 vs 0.095).
6. **Weak baselines.** Private models beating ERM on AUC, ECE and Brier (Heart 0.877/0.880 vs 0.845) suggests an untuned, overfit ERM. Add logistic regression, gradient boosting, tuned weight-decay/early-stopping ERM, and report the tuning protocol.
7. **Missing fair-DP baselines.** DPSGD-F (Xu et al.), Tran et al., Esipova et al., DP + equalized-odds post-processing (Hardt), reweighting, uniform-C=C_max, a non-private class-conditional SWD, non-private Mondrian CP, and the private-quantile method of Angelopoulos et al. are cited but not compared against. It is unclear which conformal method each baseline row uses.
8. **MIA evaluation.**
   - AUC on shadow models with ~180 training points is underpowered. Carlini et al. specifically recommend **TPR at low FPR**, which is not reported.
   - Differences like 0.525 vs 0.526 vs 0.527 (Pima) are noise, and TRACER-Med's "bold" is meaningless there.
   - Lower leakage may come from regularization rather than DP. Add a regularized non-private control and a canary-based empirical-ε audit.
   - ε=15 gives almost no formal protection, so say so plainly.
9. **Fairness metrics.**
   - Thresholded SP/EO at 0.5 are sensitive to calibration. Add threshold-free measures (subgroup AUC gap, calibration within groups).
   - The AUC gap, PR-AUC, set size and rank correlation are listed in §5.3 and never reported.
   - ΔEO as a mean of gaps is non-standard (max is common). Define it or report both.
10. **Radar plot (Fig. 4).** The axes (1−MIA, 1−ECE etc.) all sit near 0.5–1, and equal weighting is arbitrary, so area is not interpretable. "Encloses traditional baselines… successful harmonization" is not supported by Table 1, where DP-EDL beats TRACER-Med on Heart. Replace it with a table or a Pareto analysis.
11. **Table conventions.**
    - The caption says bold = best among private models, but Fair-ERM's 0.090 (a non-private model) is bold in Pima ECE.
    - Bold on ties (0.003 for all three on Heart) and for differences below 1 s.d. is misleading.
    - Decimal places and ± precision vary (±0.02 vs ±0.024; "±0.00").
    - `\resizebox{\textwidth}` on 9 columns likely makes text too small.
12. **Selective reporting.** §6.5 cites infidelity only for Heart and Pima. For EHR, TRACER-Med's 0.010 is the worst except for DP-SGD, and it is omitted. The same section describes EHR attributions.
13. **Figure and text mismatch.** Fig. 2 is on EHR, but §6.3 discusses Heart coverage numbers.
14. **Hyperparameters.** σ=0.8 fixed gives ε from 2.5 to 15, so datasets are not matched on privacy. Report privacy–utility curves at fixed ε. κ, γ, λ_fair, M_proj, M_bins and the calibration split ratio are missing.

---

## 6. Minor Issues, Writing, LaTeX, Micro-Edits

### Notation collisions
- σ = noise multiplier and sigmoid σ(z). The §4.2 equation writes the sigmoid explicitly and then defines σ afterwards.
- τ is the decision threshold (§3.3) and the conformal buffer τ_a.
- **h** is the latent vector, h_θ the feature extractor, and h_{a,m} / **h**_a the histogram counts. Rename the histogram (e.g., c_{a,m}).
- 𝒟_a (a distribution in Theorem 2) vs 𝒟 (the dataset).
- e is epoch, evidence vector, and Euler's e. E is epochs vs 𝔼 expectation. S is evidence strength; 𝒮_a / 𝒮_out are sets.
- p is a prevalence p_a, a Dirichlet sample p, and p̂.
- t is the path parameter, and t_m is a bin edge, though the notation list calls t_m "grid points in G" (G is never defined elsewhere).
- α has three roles (Dirichlet, α_conf, plus "α" in text), and β three (β_e, β_conf, plus the β in the KL weight).
- a_lrelu uses the group-index letter. Use ν.
- The notation list explains a_lrelu but not q, σ, C₀, κ, γ, λ_fair, M_proj or M_bins.
- 𝒲₁ is used for both W₁ and SW₁.
- Δ₁ (sensitivity) vs Δ_SP/Δ_EO.

### Mathematical/wording details
- "u ∈ (0,1)" is correct; S>K strictly.
- "Epistemic mutual information… from observing true outcome y" is acceptable but should say nats.
- The KL regularizer description says it penalizes "non-target evidence". State that α̃ removes target-class evidence.
- Barocas/Chouldechova/Kleinberg: "predictive parity (within-group calibration)" conflates two notions. Kleinberg uses calibration within groups plus balance for the positive/negative class. Chouldechova uses PPV parity plus error-rate balance.
- The RDP statement inside the DP definition lacks "(λ, ε(λ))-RDP" phrasing. Add "[Definition: RDP]".
- "Optimal random guessing 0.500": "chance level" is better.
- The legend "TRACER-Med (Proposed)" in tables is fine.
- ΔEO is first named "Equalized Odds Disparity (Δ_EO)", then "Equalized Odds gap" in the highlights. Be consistent.
- "Singlet" should be "singleton" (§7.2).
- §3.2 says "we evaluate two standard equity metrics" but §5.3 adds the AUC gap.
- "Biomedical registries" (Trilemma intro): the data are not registries.
- §5.1 item 1 contains a sentence fragment: "OpenML ID: 53. Secondary analysis of…".
- "Three-layer architecture (hidden dims 64, 32)" is ambiguous. State layers and parameter count.
- The overused intensifiers "strictly", "exactly", "cleanly", "directly" should be reduced.
- "Premier international regulatory guidelines" is awkward.

### LaTeX / template
- `\documentclass[3p,12pt]{elsarticle}`: 3p is a production layout, and 12pt is unusual with it. Submissions normally use `preprint` or `review`. Check how `table*`/`figure*` behave after compiling.
- `hyperref` with `colorlinks=true` is fine for review. Remove it for the final production version if the journal asks.
- `proof`, `\theoremstyle` and `amsthm` can duplicate elsarticle's own loading. Check the log for warnings.
- Unused: `proposition` and several labels (`sec:method`, `sec:problem`, `sec:related`, `sec:results`, `sec:governance`). Cosmetic.
- Abstract length: roughly 330 words. Check the target journal's word cap. It is also very number-dense, and the Conclusions line carries no finding or limitation.
- The highlights are within 85 characters each, but several overclaim ("guarantees finite-sample PAC coverage", "reduces Equalized Odds gap").
- Add keywords such as "clinical decision support" and "medical AI".
- Elsevier's generative-AI declaration must **name the tool** and the purpose. The current text omits the tool, and "published article" should read "publication".
- The CRediT list claims "Data curation" for public data and is otherwise fine for a sole author.
- The Ethics statement: if MIMIC-IV/eICU summary statistics were used, state the PhysioNet credentialing and data-use-agreement compliance.
- The affiliation block lacks postcode/state fields. Add if required.
- Elsevier's data-availability wording: a persistent identifier (Zenodo DOI) is preferred over a bare GitHub URL.

### References (verify)
- Technical citations stop around 2022 (gibbs2023, fda2024, euaiact2024 are the exceptions). With today's date at 2026-10, add 2023–2026 work on fair DP, private conformal prediction, evidential-learning critiques, and Mondrian CP. Also add Papernot & Steinke on private hyperparameter tuning, and auditing papers (Jagielski, Nasr).
- `patel2022towards` is cited both for clinician explainability (Intro) and for empirical privacy auditing (§2.4). Check that it supports both.
- `esipova2022gradient`: the ICLR venue year is probably 2023. Check.
- `romano2019conformalized` (CQR) is cited as group-conditional CP. It is not. Cite Mondrian/class-conditional CP instead.
- Check all `\cite` keys resolve and DOIs are present after a compile.

---

## 7. Final Verdict & Checklist of Actionable Revisions

**Recommendation: Major revision, bordering on reject-and-resubmit.** The core privacy and coverage guarantees need re-derivation, and the empirical claims need re-running and re-framing.

### Must fix before further review
1. **Theorem 1 / CC-SWD.** Make the alignment term per-sample (or privatize its statistics), or bound the cross-record influence. Otherwise remove the DP claim for the SWD path and the "exact / no leakage" language.
2. **Benchmark EF-SC-DP against uniform clipping at C=C_max**, plus DPSGD-F and Tran et al. Report per-group gradient norms, clip fractions and evidence. If the benefit disappears, reframe the contribution.
3. **Theorem 2 privacy.** Privatize n_a (or change the adjacency), and drop or re-account the marginal-threshold shrinkage.
4. **Theorem 2 PAC.** Bound the cumulative noise (≈M× or a CDF-level tail bound), and handle the max(0,·) bias and the noisy-total normalization.
5. **Report calibration split sizes, M_bins, average set size and marginal coverage.** Show whether the theorem's rule gives trivial sets. Add Mondrian (a,y) calibration.
6. **Reconcile all numbers** across the abstract, §6, §7.1, the conclusion and Table 1 from one results file. Remove the "42% EO / 27% SP" claims unless Table 1 supports them.
7. **Rewrite the results narrative honestly.** DP-SGD reduces disparity vs ERM here. DP-EDL beats TRACER-Med on Heart on most metrics. EHR shows no benefit. EO is worse on Heart. The ablation shows trade-offs, and the ECE claim fails.
8. **Redo the statistics.** Use repeated CV with CIs and paired tests. Fix the ablation (a post-hoc module cannot change AUC/ECE) and check for result reuse.
9. **Replace or augment the simulated EHR** with real data, or recalibrate prevalence (5–15%), the feature set and the outcome definition. Specify the protected attribute and the disparity mechanism.
10. **Privacy reporting.** Give privacy–utility curves at fixed ε≤8 (and ε≈1), account for hyperparameter tuning and for S̄_a/p_a/baselines, state the overall (ε, δ) for model plus conformal, and report MIA TPR@low-FPR with a canary audit.

### Should fix
11. Add the missing baselines: tuned LR/GBM, tuned ERM, non-private class-conditional SWD, DP + post-processing, and private-quantile CP.
12. Report the metrics promised in §5.3 (PR-AUC, subgroup AUC gap, set size, faithfulness correlation), or delete them.
13. Validate Channel B on experiments with known epistemic structure, and fix the categorical-baseline claim (one-hot encode or redefine).
14. Justify the u<0.20 threshold with selective-risk and abstention-by-group analysis, or soften the workflow section.
15. Report SP against the base-rate (oracle) SP. Drop SP as a headline.
16. Replace the radar chart with a table or Pareto analysis.
17. Moderate the regulatory text. Replace "supports compliance" with "supports documentation of", fix the PCCP title, and update the status of the EU AI Act and HHS provisions.
18. Rename "Fisher/curvature" and "Transport" to what they are, or derive them.
19. Update the references (2023–2026 work) and correct the citation roles flagged above.

### Cosmetic
20. Resolve the notation collisions, fix the table formatting and bold conventions, fix the generative-AI declaration, trim the abstract, and clean up wording and LaTeX options.

I can also mark up a corrected Theorem 2 (with the cumulative-noise bound) or draft a per-sample private alignment mechanism for CC-SWD if you want.
