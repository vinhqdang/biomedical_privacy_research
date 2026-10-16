# Referee Report: TRACER-Med (Artificial Intelligence in Medicine)

I reviewed the LaTeX source only. I did not see `references.bib`, the figures, or the code, and I did not recompute the RDP accountant. Items that depend on those are marked "verify".

---

## 1. Overall Assessment & Summary of Contributions

The manuscript combines four components in one pipeline for tabular clinical classification:
- an evidential Dirichlet MLP;
- DP-SGD with group-dependent clipping thresholds (SC-DP-SGD);
- a DKW-buffered, Laplace-perturbed group-conditional conformal threshold (DP-GCCP);
- two-channel path-integrated attributions (DCDA).

It is evaluated on Statlog Heart (N=270), Pima (N=768) and a synthetic EHR cohort (N=10,000).

The topic is timely and the framing is clear. The integration is a reasonable engineering idea.

In its current form the paper's central claims are not supported. They are:
- (a) a "Trilemma" is formalized and mitigated;
- (b) SC-DP-SGD reduces disparity;
- (c) DP-GCCP gives equitable private coverage;
- (d) MIA results corroborate the privacy guarantee;
- (e) DCDA gives faithful dual explanations.

The main problems:
- The privacy "theorem" is a restatement of standard DP-SGD with clip `C_max`.
- The conformal quantile formula contains no randomness, so as written it is not DP.
- The ablation table contradicts the paper's own description of the method.
- The experiments are statistically underpowered and use weak or inappropriate baselines.
- The EHR benchmark is degenerate.
- Several numbers are internally inconsistent.

**Recommendation: major revision.** The paper is not publishable until the issues in §3 and §5 are fixed.

---

## 2. Major Strengths

1. The problem is well motivated and the four-pillar framing suits a clinical audience.
2. The author correctly adopts add/remove adjacency, Poisson-subsampling-based RDP accounting, and a split-level composition argument.
3. The author honestly acknowledges the Barocas/Chouldechova/Kleinberg impossibility results and makes Equalized Odds primary.
4. The author is transparent about using population priors rather than batch counts for the weights, in principle.
5. The Dirichlet/EDL identities are correct (see §3.1).
6. Decoupling the batch-coupled SWD from the private step is a correct observation.
7. The limitations section is candid about binary attributes, synthetic data and static data.
8. Code is promised. Regulatory references are specific and mostly accurate (FDA GMLP, PCCP final guidance of Dec 2024, §92.210, AI Act Articles 10/13/14/15).

---

## 3. Critical Methodological & Mathematical Issues

### 3.1 Evidential Dirichlet formulas (mostly correct)
- Verified correct:
  - `α_k = e_k+1`, `p̂_k = α_k/S`, `u = K/S`;
  - expected entropy `Σ p̂_k(ψ(S+1) − ψ(α_k+1))`;
  - mutual information as `H(p̂)` minus expected entropy;
  - the Sensoy Bayes-risk loss and the annealing `λ_t = min(1, t/10)`.
- `u ∈ (0,1]` should be `(0,1)`, because softplus is strictly positive.
- The KL term is never written out. Give the closed form for `KL(Dir(α̃) ‖ Dir(1))`.
- **Conceptual overclaim.** Vacuity `K/S` is not Bayesian epistemic uncertainty. Bengs et al. 2022 (already cited) and Jürgens et al. (ICML 2024) show that EDL does not identify epistemic uncertainty faithfully.
  - Nothing here shows `u` is informative under DP noise. There is no OOD detection, no selective-prediction/AURC analysis, and no validation of `u<0.20`.
- The `+1` prior shrinks `p̂` toward uniform. Some of the ECE/Brier gain over softmax models is a built-in shrinkage effect, not "calibrated uncertainty". Add a temperature-scaled DP baseline.

### 3.2 Privacy theorem (SC-DP-SGD)
1. **The theorem is trivial.**
   - Clipping to `C_a ≤ C_max` and adding `N(0, σ²C_max²I)` is exactly standard DP-SGD with clip `C_max` and multiplier `σ`.
   - The proof cites the subsampled-Gaussian RDP and adds nothing new.
   - It is therefore also identical to uniform clipping in accounting. The only novelty is a static per-group rescaling of gradients, closer to cost-sensitive reweighting than to "sensitivity calibration".
   - "Dynamic sensitivity reweighting" in the abstract is wrong, because the weights are static.
2. **Numerics inside a theorem.** Hyperparameters and ε values do not belong in a theorem statement. Move them to a table.
3. **T is inconsistent with its own formula.**
   - The theorem defines `T = E·⌈N_tr/|B|⌉`. That gives 120 (Heart) and 636 (EHR), not the reported 113 and 625.
   - The reported values equal `E·N/|B|`, i.e. Opacus-style `E/q`. Fix the formula or the accounting.
4. **Poisson sampling is assumed but not shown.** Algorithm 1 says "mini-batch ⊂ D_tr" and the setup says "batch size 32". State that Poisson sampling was implemented, and that normalization uses the expected batch size.
5. **The update rule disagrees.** Algorithm 1 uses SGD (`θ ← θ − ηG̃`), but the experiments use Adam. Adam is fine as post-processing, but say so.
6. **ε = 15.36 is not "provable patient privacy".** It is a weak guarantee. Report the baseline ε alongside a stronger setting (ε ≤ 8, ideally ≤ 1–3). Note that the same σ yields ε from 2.5 to 15.4 across datasets.
7. **Unaccounted data-dependent steps.** These are outside the stated ε:
   - within-fold median imputation;
   - feature scaling;
   - hyperparameter selection (lr, σ, γ, epochs);
   - the `1.17σ` / weight settings, if tuned on data.
8. **The "fixed external priors" claim is contradicted by the numbers.**
   - The weights use 32%, 29% and 46%, which are the in-sample proportions. No registry figure is cited for any of them.
   - There is no registry prior for "Pima age ≤30" or for a synthetic "demographic cohort".
   - If in-sample frequencies are used, the "zero auxiliary leakage" claim fails.
9. **The majority noise multiplier is wrong.**
   - For γ=0.5, `w̃_a = p_a^{-1/2}/√2`, so `w̃_0 = 1.25` and `w̃_1 = 0.858`.
   - `C_max/C_maj = √(0.68/0.32) = 1.46`, not `≈1.17`. (The "1.17" corresponds to γ ≈ 0.21.)
10. **Confounded comparison.**
    - SC-DP-SGD injects noise `σ·1.25·C`, while the DP-SGD baseline uses `σ·C`.
    - Compare against uniform clipping at `C_max`, and against DP loss-reweighting.
11. **Minor accounting points.**
    - Wang et al. 2019 covers sampling without replacement; Mironov 2019 covers Poisson.
    - Integer orders up to 64 lose tightness. Use the Balle et al. 2020 conversion or a PRV accountant.
    - Define whether `ε(λ)` in the theorem is per-step or cumulative.
    - Verify 15.36/6.13/2.49 with an independent accountant.

### 3.3 DP conformal quantile (Eq. `dp_quantile`) and Proposition 2
1. **The formula is deterministic.**
   - `q̂_a = inf{t : F_S(t) ≥ 1−α+τ_a+ln(4/β)/(|S|ε)}` contains no noise. The text says the index is "perturbed with Laplace noise", but no `ξ_a` appears.
   - Releasing this exactly is not ε-DP.
   - You need a noisy CDF over a public grid (as in Angelopoulos et al.), an exponential mechanism, or a noisy binary search. Then the buffer is a high-probability correction for that noise.
2. **Sensitivity.**
   - "Global ℓ₁ sensitivity of the CDF = 1/|S_a|" is only valid for a single fixed threshold query.
   - The denominator `|S_a|` is itself data-dependent under add/remove. State it as `1/(|S_a|+1)` or use counts.
   - `|S_a|` is then released through `τ_a`, and that release is not accounted for.
3. **Vacuity at small n.**
   - With β=0.05 and n=15–20, `τ_a = 0.38–0.33`, so the target level is `0.9+τ_a > 1`.
   - The threshold is then trivially 1 and every set is `{0,1}`.
   - This explains the 97.1% marginal coverage, the set size of 1.84, and the "w/o SWD" gap of exactly 0.000±0.00.
   - It is not "sharp coverage"; at n=2,000 the buffer is still 0.033, so `τ_a → 0` is overstated.
4. **Proposition statement.**
   - It claims `P(Y ∈ C(X) | A=a) ≥ 1−α`. The proof gives a PAC (training-conditional) statement: `F_a(q̂_a) ≥ 1−α` with probability ≥ 1−β over the calibration sample.
   - Exchangeability is not what is used. The DKW argument needs i.i.d. scores.
   - Rewrite precisely and state the conditioning.
5. **Constants.**
   - The Laplace tail `½e^{-x/b}` at `x = b·ln(4/β)` gives `β/8`. You then spend `β/2` on it.
   - The buffer could be `b·ln(1/β)`. This is sloppy, not wrong.
6. **Define `inf ∅`.** Set `q̂=1`.
7. **Novelty.** The contribution over Angelopoulos et al. "Private Prediction Sets" is a per-group application. State the difference and compare against it empirically.
8. **Privacy composition.** "max(ε_model, ε_conf)" holds under parallel composition only if the train/cal split is data-independent. Say so. The abstract's ε omits the 1.0.
9. **Use of the protected attribute at inference.** The model excludes sex on Heart but the conformal step requires it. This is internally inconsistent and legally sensitive (see §4.5).

### 3.4 Sliced Wasserstein alignment
- The integral of `|F₀⁻¹ − F₁⁻¹|` is the sliced W₁, not W₂. Fine, but state it.
- It aligns marginals `P(h|A)`, which targets demographic parity. The paper argues SP is clinically inappropriate. Use class-conditional `P(h|A,Y)` alignment to match Equalized Odds.
- SWD is applied "on public/unlabelled reference data" that does not exist for any of the three benchmarks.
  - If it is drawn from the same dataset, the DP claim is violated.
  - Group labels are also required.
  - Specify the data and the protocol.
- If SWD is absent from private training, it cannot contribute to the TRACER-Med results (see §5.3).

### 3.5 DCDA
- Channel A attributes `ln α_ŷ`. This is evidence for the predicted class, not risk. A feature that raises evidence for both classes gets positive attribution. Attribute `p̂_ŷ` or the log-odds.
- Channel B: IG completeness gives `Σ_j Φ_j = u(x) − u(x⁰)`. Since `u(x⁰) ≠ 0`, these are changes relative to the baseline, not "drivers of uncertainty at x".
- Interpolating categorical or ordinal inputs (chest pain type, Charlson, cohort indicator) is meaningless.
- A single "healthy" reference point (BP 120, glucose 90) lies off-manifold. Heart rate, WBC and creatinine baselines are not given.
- Channel B's claim of "missing or borderline indicators" is unsupported. The model has no missingness mechanism.
- "Gauss–Legendre with m=50 integration steps" should read nodes.
- No attribution-stability evaluation under DP is given, although the introduction's fourth bullet claims DP destabilizes attributions.
- Planned metrics (faithfulness rank correlation) are never reported. See §5.4 for the infidelity numbers.

---

## 4. Clinical & Biomedical Accuracy

1. **Synthetic EHR cohort.** The generator, correlation structure, missingness, time window, units and outcome prevalence are not described.
   - Mixing "30-day deterioration" with "ICU mortality" is a conflation.
2. **Feature problems.**
   - SOFA is derived from creatinine, platelets, MAP, bilirubin, PaO₂/FiO₂ and GCS. Several of these are absent, and some are duplicated as separate inputs. This is redundancy or leakage.
   - SpO₂ is uninterpretable without FiO₂.
   - SBP is used where MAP is standard. The hypotension threshold is usually SBP<90, not <85.
   - Lactate > 4 mmol/L is a reasonable severe cutoff.
   - The cohort indicator, which is the protected attribute, is included as an input here but excluded on Heart.
3. **The model learned almost nothing on EHR.**
   - AUC 0.67–0.68 is far below real ICU mortality models (≈0.8–0.9).
   - Brier 0.076–0.078 equals the constant-predictor Brier if prevalence is ≈8% (verify; prevalence is not reported).
   - At τ=0.5 with ~8% prevalence and this AUC, almost nothing is flagged positive. That is why ΔSP and ΔEO are `<0.001`.
   - The `ΔEO < 0.001` cited in §7 as evidence of equity is an artifact of a null simulation and a degenerate classifier.
   - The DCDA "clinical" findings (lactate, hypotension, SOFA) recover relationships built into the simulator. This is circular and has no clinician evaluation.
4. **Real-data limitations.**
   - Statlog Heart (N=270) is the 1988 Cleveland referral cohort, selected for angiography.
   - Pima is a single-community, all-female dataset, with ethical/data-sovereignty considerations beyond the Ethics Statement.
   - D mismatch: the Pima feature list names 7 features while claiming D=8. Either age (the protected attribute) is included or D=7.
   - Verify the group split. I recall the median Pima age as ≈29, which would make "≤30" the majority group (≈54%), not 46%. The labels may be swapped.
5. **Protected attributes are also clinical risk factors.**
   - Sex and age are legitimate predictors of CAD and diabetes. Equalizing positive rates or even error rates across them is not obviously clinically desirable.
   - Excluding sex from the features is fairness through unawareness, yet sex is used at inference for the thresholds.
   - HHS §92.210 requires covered entities to identify tools using protected inputs such as sex and age. Sex-conditioned conformal thresholds are therefore in scope. Check the current enforcement status.
6. **Clinical workflow.**
   - `u<0.20` for K=2 means `S>10`. This is arbitrary and unvalidated.
   - The abstain-and-route policy is not evaluated: no abstention rate, no accuracy on retained cases, no alert burden.
   - A 0.5 threshold is inappropriate for an 8%-prevalence outcome. Use decision-curve analysis or operating points fixed by sensitivity/PPV.
   - `{0,1}` sets for 84% of Heart patients (set size 1.84) means the "co-pilot" abstains almost always.
7. **Regulatory section.**
   - GMLP Principle 3 concerns dataset representativeness. SC-DP-SGD is algorithmic and does not satisfy it.
   - Principle 10 concerns monitoring and retraining risk, not privacy.
   - The PCCP paragraph ("guardrails for continuous retraining without leakage") overreaches. Repeated retraining composes ε.
   - "Bounding demographic error disparity" is wrong. These are empirical estimates, not bounds.
   - Add AI Act Articles 9, 11 and 12, and Art. 10(5) for bias detection with sensitive data. Check the current application timeline for Annex I high-risk obligations, which have been subject to revision proposals.

---

## 5. Statistical & Experimental Soundness

### 5.1 Power and uncertainty
- With 3 folds × 3 seeds, n=3. Standard deviations on 3 values are not meaningful.
- Heart test folds have 90 patients, about 29 of them female, and roughly 10 diseased females. ΔEO differences of 0.05–0.06 are well within noise.
- There are no CIs, no paired tests and no multiplicity control. Bold type is awarded without significance.
- Use repeated stratified CV (≥10×5), bootstrap CIs over test sets, and paired tests.
- ECE with 10 bins on n≈90 is dominated by noise. Use adaptive binning and report CIs.

### 5.2 Baselines
- The key fair-DP baselines cited in §2 are not run: DPSGD-F (Xu), Tran et al., and Esipova et al.
- Missing too:
  - DP loss reweighting;
  - group DRO or exponentiated gradient;
  - Hardt post-processing;
  - DP-logistic regression;
  - non-DP gradient-boosted trees and logistic regression, which typically reach AUC ≈0.90 on Heart and ≈0.83 on Pima.
- Fair-ERM makes fairness worse than ERM on Heart (ΔSP 0.307 vs 0.283, ΔEO 0.166 vs 0.148), so it is a weak baseline.
- DP-SGD beats ERM in AUC and ECE on Heart, which suggests the ERM baseline is untuned. No hyperparameter tuning is described for any method.
- Conformal baselines (marginal CP, Mondrian CP without DP) are not reported for the baselines in the main table.

### 5.3 The ablation table contradicts the text
- The text says "w/o DP-GCCP" is an inference-time change, so point metrics should equal the full model (0.863/0.077/0.232/0.083). The table gives 0.890/0.097/0.274/0.091.
- The text says SWD is not used in private training. The "w/o SWD" row should therefore also equal the full model, yet it differs (AUC 0.879, ΔSP 0.259, ΔEO 0.104).
- These gaps are as large as the claimed effects. They measure the run-to-run noise floor, or they reveal that SWD is in fact used. Either way, the SWD and GCCP conclusions do not stand.
- "w/o DP-GCCP" gives coverage disparity 0.046 vs 0.045 for the full model. This shows no benefit from group-conditional calibration.
  - Meanwhile the singleton rate drops from ~56% to ~16% (set size 1.44 to 1.84).
  - The text's "preventing minority undercoverage" is not shown.
- "w/o SC-DP" is better than the full model on AUC (0.885), ΔSP (0.220, bolded) and ECE. Its ΔEO of 0.177 equals the standard DP-SGD value to three decimals; check for a copy error.
- Every ablation improves AUC. The claim "each module contributes" is not established.
- Run proper factorial ablations with matched seeds.

### 5.4 Results do not support the narrative
- On Pima, DP-SGD has lower fairness gaps than TRACER-Med (ΔEO 0.216 vs 0.258, ΔSP 0.321 vs 0.361). DP-SGD also improves on non-private ERM (ΔEO 0.285→0.216).
- On Heart, DP-SGD improves calibration relative to ERM (ECE 0.199→0.146).
- Both observations undercut the "DP harms equity and calibration" premise and the claim that a Trilemma was "systematically dissected". No analysis of the Trilemma is provided.
- The 27%/42% reductions are cherry-picked against DP-EDL.
  - Versus DP-SGD, ΔSP falls only 12% (0.263→0.232).
  - The reductions are attributed to SC-DP-SGD in the contributions, but the compared system is the full pipeline, and the ablation shows w/o SC-DP has better ΔSP.
- The "superior" Pima AUC (0.826 vs 0.812, SD 0.01–0.02) is within noise.
  - The paper says clipping is "symmetric" on Pima, so TRACER-Med ≈ DP-EDL there.
  - The claimed improvement then cannot come from the proposed mechanism.
- Coverage gap:
  - Standard DP-SGD has the best gap on all three datasets (0.001, 0.006, 0.009).
  - TRACER-Med on Pima has 0.058, ten times worse than DP-SGD and DP-EDL.
  - On Heart its gap is 0.045±0.05, indistinguishable from 0.
  - On EHR all methods are at 0.009 (a noise floor), so Fig. 2 cannot show a benefit.
- The set-size sharpening (1.84 → 1.12) compares different tasks. The 88% singletons on EHR follow from ~92% majority class, not from model quality.
- Infidelity: TRACER-Med is worse than DP-SGD and Fair-ERM on Heart (0.015 vs 0.005/0.004). The comparison is also unclear:
  - Which attribution method is used for non-evidential models?
  - Infidelity depends on output scale.
  - "Stable attributions despite DP training" is unsupported.
  - The infidelity sentence for Heart/Pima sits under inpatient attributions.
- MIA:
  - AUC ≈ 0.5 with ε=15 does not "corroborate" the bound. MIA gives only weak lower bounds.
  - On EHR, even non-private ERM is 0.514, so there is nothing to detect.
  - Use LiRA-style TPR at low FPR, canary auditing, and a per-group MIA analysis (disparate privacy risk). Carlini 2022 does not cover "DP biomedical classifiers" specifically.
- Metrics promised but not reported: PR-AUC, Subgroup AUC Gap, Faithfulness Rank Correlation, set size for baselines.
- Fig. 4 radar:
  - "1 − MIA AUC" is maximal at MIA=0, but the optimum is 0.5. The transform is wrong.
  - Radar areas on compressed 0.8–1.0 axes are not informative.
  - "Encloses single-objective baselines" is contradicted by the table.
- Fig. 1's caption claims a privacy trade-off, but privacy is not an axis. It plots ΔSP, though ΔEO is declared primary.

---

## 6. Minor Issues, Writing, LaTeX, Micro-Edits

**Front matter and metadata**
1. The title in the source differs from the requested title ("Harmonizing Privacy, Fairness, and Uncertainty…" vs "A Unified Trustworthy Deep Learning Framework…"). Reconcile.
2. Highlights: several exceed Elsevier's 85-character limit (e.g. "Unifies differential privacy, subgroup fairness, evidential uncertainty, and explainability."). They should be plain text, but they contain `$C_{\max}$` and `R{\'e}nyi`. "Provable" is an overclaim.
3. The abstract is overlong and number-dense (check the AIiM limit). Reduce.
4. Abstract: "dynamic sensitivity reweighting" is incorrect (static). "Corroborated by resistance to MIA" is overstated. "Superior … calibration (Brier 0.173)" conflicts with Heart results where DP-EDL has the better Brier.
5. TRACER acronym: "Aligned" is not a delivered private component, and explainability is absent from the acronym.
6. AI-use statement: Elsevier requires the tool name. Because AI was used for "literature cross-referencing", verify every reference exists and is correctly attributed.

**Terminology and notation**
7. "Trilemma" refers to three pillars but the paper lists four pillars. Align.
8. `α` is the Dirichlet parameter, `α_conf` and the LeakyReLU slope. `τ` is the decision threshold and the DKW buffer. `t` is epoch, CDF argument and path parameter. `S` is evidence strength and `𝒮_a`/`𝒮` event sets. `E` is epochs and expectation. `𝒟` is the dataset and `𝒟_KL`. `m` is the quadrature node count and `M` the projection count. `λ_t` and `λ_RDP`. Use distinct symbols.
9. `p_a` (prior), `p̂_k`, `P`, and `q` (sampling ratio) vs `q̂_a`. Define once.
10. "Statistical Parity Disparity (Δ_SP)": consider SPD/DPD. "Predictive parity (within-group calibration)" conflates two notions.
11. "Singlet prediction set" should be "singleton". "Informative-ness" should be "informativeness". "Non-conformity"/"nonconformity", "unlabelled" (British) vs American spellings elsewhere.
12. Mixed `R{\'e}nyi` and `R\'enyi`.
13. Equations lack terminal punctuation.
14. The `1/2` factor in `w_a` cancels under normalization and is redundant.
15. "p_a < 0.5 ⟹ w̃_a > 1" holds only for γ=0.5 under this normalization.
16. "3-layer MLP (64, 32)" is ambiguous. State d and the head.
17. "Heart D=12 … etc." should not end with "etc." in a data specification.

**Citations**
18. `kaissis2020secure` does not support "diagnostic accuracy comparable to specialists".
19. `abadi2016deep` is cited for the definition of DP and for DP in general; cite Dwork et al. instead.
20. `romano2019conformalized` is CQR (regression), not group-conditional classification.
21. Check `patel2022towards` against the claims it supports.
22. The references stop at 2024. Add recent work on DP and fairness in medical imaging (e.g. Tayebi Arasteh et al. 2024, Commun. Med.), evidential-learning critiques (Jürgens 2024), DP conformal methods, and fair DP-SGD from 2023–2026. This follows your instruction on current references.

**LaTeX**
23. `[3p,12pt]`: 12pt with `3p` may warn. A 9-column `\resizebox{\textwidth}` table inside `\small` renders at tiny font size. Split or rotate it.
24. `algorithm` + `algorithmic` is dated; consider `algpseudocode`.
25. Load `hyperref` last and check clashes with elsarticle's own theorem definitions.
26. `\affiliation[...]` and `highlights` require a recent elsarticle; verify the build.
27. Table caption says "3 seeds" while the text says "3-fold CV with seeds". Define.
28. Entries `<0.001` have no SD. Decide on a convention.
29. Table 2 caption "Bold values denote the best performance for each metric" but several bolds are ties or noise-level differences.
30. Algorithm 1: add SWD, KL annealing, Poisson sampling, DCDA, and how σ is derived from `(ε, δ)`. At the moment ε is an input that is never used.

**Ethics and availability**
31. State that no credentialed MIMIC-IV/eICU access was used. Add Indigenous-data considerations for Pima.
32. Describe the synthetic-data generator and include its seed.

---

## 7. Final Verdict & Checklist of Actionable Revisions

**Verdict: major revision.** The integrated design is interesting, but the privacy contribution is trivial, the private conformal procedure is incorrectly specified, and the experiments do not support the claimed benefits.

**Must fix (blocking)**
1. Correct Eq. `dp_quantile`: add an actual DP mechanism (noisy CDF on a grid, exponential mechanism or noisy binary search), derive its sensitivity properly, and restate the Proposition as a PAC (training-conditional) guarantee. Compare to Angelopoulos et al.
2. Resolve DKW vacuity for n ≈ 15–20: either report that sets are trivial, or use a less conservative construction, or abandon group-conditional guarantees on tiny cohorts.
3. Resolve the ablation contradictions (§5.3) with matched-seed factorial runs. Either SWD is in the private model or it is not. Make the claims consistent.
4. Fix the `1.17σ` figure (should be ≈1.46 at γ=0.5). Fix `T` vs `⌈·⌉`. State and verify Poisson sampling. Recompute and verify ε with an independent accountant.
5. Reframe the "theorem" as a corollary of standard DP-SGD. Drop "dynamic", "derive closed-form fairness bounds" and "prove" claims that are not proven.
6. Replace the EHR benchmark with a real, credentialed dataset (MIMIC-IV, or at least the public eICU demo) or a simulator with group-dependent signal. Report prevalence and a clinically sensible operating point. Remove the `ΔEO<0.001` regulatory claim.
7. Add the missing fair-DP baselines (DPSGD-F, Tran et al., DP-reweighting), GBM/logistic baselines, and tuning protocols with accounting of tuning leakage. Make the "uniform clip at `C_max`" control.
8. Report statistics properly: ≥10 repeats, bootstrap CIs, paired tests. Drop bolding unless significant.
9. Replace "MIA corroborates formal ε" with an honest statement. Add low-FPR LiRA/canary audits and per-group MIA.
10. Report ε for all datasets in the abstract alongside ε_conf, and discuss ε=15.36 honestly.

**Should fix**
11. Correct DCDA channel definitions (attribute probability or log-odds; clarify the Channel B baseline offset). Handle categorical features. Provide stability and faithfulness evaluations, plus a clinician or reader assessment.
12. Validate `u` thresholds and abstention via selective-prediction curves, OOD detection, and decision-curve analysis.
13. Clarify the use of protected attributes at inference; discuss §92.210 and AI Act Art. 10(5).
14. Align the SWD to class-conditional if Equalized Odds is primary.
15. Rewrite the regulatory section more conservatively (GMLP 3 and 10 mappings, PCCP).
16. Verify Pima D and the age-group proportions; resolve the sex/protected-attribute inclusion inconsistency between Heart and EHR.
17. Update the references to 2025–2026 and verify every bibliography entry (given the AI-assisted cross-referencing).
18. Fix notation collisions, highlights length, abstract length, figure captions (Fig. 1, Fig. 4 transform), and the LaTeX items in §6.

I can turn this into a point-by-point revision plan or check specific numbers against the repo's code and results if you want. Tell me which.
