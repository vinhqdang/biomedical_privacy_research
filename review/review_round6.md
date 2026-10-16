# Peer Review: TRACER-Med (*Artificial Intelligence in Medicine*)

I reviewed this from the LaTeX source only. I did not run the code or open the figures. Figures 1–4 are assessed from their captions and the surrounding text. Severity tags: **[C]** critical (a claim is false or unsupported as written), **[M]** major, **[m]** minor.

---

## 1. Overall Assessment & Summary of Contributions

The manuscript proposes TRACER-Med, which stacks five components:
- an evidential Dirichlet network;
- group-adaptive DP-SGD clipping (EF-SC-DP);
- a class-conditional latent alignment penalty (CC-SWD);
- a Laplace-based private group-conditional conformal calibrator (DP-GCCP);
- a two-channel integrated-gradients explainer (DCDA).

It evaluates the stack on two tiny public tabular datasets (Statlog Heart, N=270; Pima, N=768) and one author-simulated EHR cohort.

The problem is timely and the ambition is appropriate for the journal. The paper reads fluently.

**The central claims are not supported by the theory or the evidence as written.**
- The privacy proof of the conformal module is incorrect (sensitivity is up to M, not 1).
- The RDP bound is applied outside its validity regime (σ=0.8).
- The "CC-SWD" that is actually trained is not a Wasserstein distance.
- The "Fisher-derived" clipping rule contains no Fisher quantity and appears to push in the opposite direction from its motivation.
- The empirical results do not show the claimed advantage. TRACER-Med wins roughly 2 of 8 metrics on Heart. It is worse than DP-EDL on fairness on Pima. On the EHR cohort all methods tie at zero disparity.
- The ablation contradicts several of its own conclusions.

**Recommendation: major revision, bordering on reject-and-resubmit.** The theory needs correction and the experiments need redesign.

---

## 2. Major Strengths

1. The motivation is clinically well chosen. Disparate impact of DP, calibration, and conformal coverage interact in practice, and treating them jointly is a worthwhile direction.
2. Preferring Equalized Odds over Statistical Parity under unequal base rates is the right clinical call.
3. The group-conditional DP conformal idea (a public grid with calibration kept disjoint from training) is a sensible architecture.
4. The use of LayerNorm rather than BatchNorm is DP-compatible. Add/remove adjacency, Poisson sampling, and post-processing via Adam are stated correctly.
5. Notation conventions are declared up front, which is unusual and welcome.
6. Several reported percentages check out arithmetically: 42%, 53%, 27%, and 113%.
7. The limitations section acknowledges the synthetic cohort and binary-attribute restrictions.
8. Code and data availability are stated.

---

## 3. Critical Methodological & Mathematical Inquiries

### 3.1 Dirichlet / EDL formulas (verified)
Correct:
- Expected aleatoric entropy, `−Σ p̂_k(ψ(α_k+1) − ψ(S+1))`.
- The KL term against Dir(1), including α̃ = y + (1−y)⊙α.
- The Dirichlet FIM, `g_jk = ψ₁(α_j)δ_jk − ψ₁(S)`.
- The bound `u ∈ (0,1)`, which holds because softplus > 0 strictly.

Problems:
- **[C] The gradient formula in §4.2 is wrong.** It reads `∂L_Bayes/∂z_k = (ψ(S)−ψ(α_k))·e^{z_k}/(1+e^{−z_k})`.
  - The softplus derivative is the sigmoid `1/(1+e^{−z})`. The extra `e^{z_k}` factor is spurious.
  - For `L = ψ(S) − ψ(α_c)`, `∂L/∂α_j = ψ₁(S) − δ_{jc}ψ₁(α_c)`. The gradient involves trigamma, not digamma differences.
  - The Introduction repeats the error ("loss gradients depend on digamma differences"). The loss value does; the gradient does not.
  - The qualitative story ("low evidence gives large gradient norm") may still hold through ψ₁, but the displayed derivation cannot be used as written.
- **[m]** `L_Bayes` is the expected log-loss (adjusted cross-entropy) of Sensoy et al. Their "Bayes risk" is the squared-error variant. Rename it.
- **[m]** "Mutual information … from observing true outcome y" should be "a label sampled from the predictive distribution".

### 3.2 EF-SC-DP: the "Fisher" derivation is not a derivation
- **[C]** No Fisher quantity appears in `w_a`. It contains only `S̄_a^κ` and `(1/2p_a)^γ`.
  - "Derive … based on the Fisher Information Metric" is an overclaim.
  - "Natural gradient norm scales with evidence strength" is asserted, not shown.
  - "Fisher-weighted SNR (F-SNR)" is never defined, and the claim of equalization is never proven.
- **[C] The direction of the S̄ term contradicts the motivation.**
  - The text says low evidence produces large gradients, so minority gradients are over-truncated and need a larger threshold.
  - But `w_a ∝ (S̄_a/Σp S̄)^κ` with κ>0 gives the *lower*-evidence group a *smaller* threshold. This opposes the `(1/2p_a)^γ` term.
  - Either the exponent should be −κ or the narrative is inverted.
- **[M] SNR cannot be equalized this way.** Noise is scaled to the common `C_max`, and the signal per group is bounded by `C_a`. Group SNR is therefore `≤ C_a/(σC_max) ≤ 1/σ`. Differing thresholds can only lower the SNR of the non-max group relative to its own threshold.
- **[M] The normalization `Σ p w̃² = 1` implies `max w̃ ≥ 1`.** Hence `C_max ≥ C_0`, so TRACER-Med injects at least as much absolute noise as DP-EDL at the same σ. Report effective noise, clipping fractions per group, and gradient-norm distributions per group. The central premise (minority gradients are larger or heavier-tailed) is never shown empirically.
- **[M] Provenance of `p_a` and `S̄_a`.**
  - The claim that `p_a` comes from an "external registry" and has "zero privacy leakage" is not credible here. The 32.2% female share equals the empirical Statlog proportion. The AHA registry does not give it. NHANES is irrelevant to Pima, a specific population of women aged ≥21, so Pima's `p_a` cannot come from NHANES either.
  - If `p_a` or `S̄_a` was set from the training data, privacy accounting is violated.
  - `κ, γ` are never reported (see §5).

### 3.3 Theorem 1 (RDP)
- **[C] The per-step bound is applied outside its regime.**
  - `ε_step(λ) ≤ q²λ/(2σ²) + O(q³)` is a small-q, large-σ approximation. Abadi's lemma requires σ ≥ 1 and λ ≤ σ²ln(1/(qσ)).
  - The experiments use σ=0.8 and q = 32/≈180 ≈ 0.18 on Heart. The leading term is actually ≈ (λ/2)q²(e^{1/σ²}−1), roughly 2.4× larger than the stated one at σ=0.8, and the O(q³) remainder is not negligible at q=0.18.
  - The theorem writes "≤ … + O(q³)" (not a rigorous inequality), while the proof drops the remainder and states a clean "≤". These are inconsistent.
  - The reported ε (15.36, 6.13, 2.49) presumably came from an exact numerical accountant. Name it (Opacus RDP/PRV?). The formula in the theorem is not what produced those numbers.
- **[M]** "Exact Rényi DP" (abstract) is not accurate. Use the Mironov et al. numerical bound or a PRV accountant. Optionally use the tighter RDP-to-DP conversion of Balle et al. (2020).
- **[C] Algorithm 1 divides by `|B|`.** With Poisson sampling, `|B|` is a data-dependent random variable and the division leaks. Use the expected batch size `qN`.
- **[M] CC-SWD as implemented may violate the sensitivity argument** (see 3.4).
- **[M] Unaccounted data-dependent quantities.** These include feature standardization, imputation, `τ̄_y`, `S̄_a`, `s_a`, DCDA medians, hyperparameter tuning, and early-stopping choices.
- **[M]** An ε ≈ 15 on N≈180 is not a meaningful privacy guarantee. A near-chance MIA does not "corroborate" the bound (see §5).

### 3.4 CC-SWD
- **[C] The trained loss is not CC-SWD.**
  - The stated CC-SWD is a sliced W₁ between latent distributions `P(h|A=0,Y=y)` and `P(h|A=1,Y=y)`.
  - The "per-sample gradient" is `∇_θ[s_{a_i}(p̂₁(x_i) − τ̄_{y_i})²]`. That is a squared deviation of the *output probability* from a class constant, weighted by group. It does not involve `h`, quantiles, or any cross-group comparison.
  - If `τ̄_y` is a class-wise constant, this is group-reweighted MSE rather than distribution alignment. The "w/o SWD" ablation may simply be measuring reweighting.
  - A true sliced-Wasserstein term depends on batch quantiles, so it is not per-sample decomposable. That undermines both "per-sample clipping" and "no auxiliary privacy expenditure".
  - `τ̄_y` and `s_a` are undefined. If `τ̄_y` is computed from the data, record `i`'s gradient depends on other records and the `C_max` sensitivity proof fails.
- **[m] Notation collisions.** `θ_m` (projection direction) clashes with `θ` (parameters). `M` is SWD projections, grid size, and quadrature nodes. `u` is the vacuity and also an integration variable.
- **[m]** The sliced W₁ should be `E_{θ∼Unif(S^{d−1})}[·]`, approximated by M samples.
- Fair-ERM uses a (presumably marginal) SWD. State which. The proposed method's key difference (class-conditional) should be isolated with marginal-SWD and class-conditional-SWD baselines.

### 3.5 DP-GCCP (Theorem 2)
- **[C] The privacy proof is incorrect.**
  - Adding one record with score `s` increments `c_a(t_m)` for *every* `t_m ≥ s`, up to M=100 coordinates.
  - The L1 sensitivity of the query vector is up to M, not 1. Per-coordinate `Lap(1/ε_conf)` noise gives roughly `M·ε_conf`-DP, up to ≈100 for M=100, not `ε_conf`.
  - Fixes:
    - Laplace noise of scale `M/ε`, which makes the buffer ~100× larger.
    - A histogram or binary-tree mechanism with cumulative sums as post-processing.
    - The exponential mechanism or the private quantile of Angelopoulos et al.
  - The Highlights ("guarantees finite-sample PAC coverage") and the "ε_conf = 1.0" in the abstract are affected.
- **[M] `n_a` is used non-privately.** `F̃ = (c+ξ)/n_a` and `τ_a` both use the exact group count. Under add/remove adjacency `n_a` is itself sensitive. Treat it as public or release it noisily.
- **[C] The theorem is vacuous at the dataset sizes used.**
  - The target is `1 − α + τ_a`.
  - For n_a=30: τ ≈ √(ln80/60) + ln(4000)/30 ≈ 0.27 + 0.28 = 0.55. For n_a=100: ≈ 0.23.
  - Target > 1, so `inf ∅ = 1` and all prediction sets are `{0,1}`. Coverage is then 100% trivially. This probably explains the 97.1% marginal coverage at 90% nominal on Heart, and the "0.000 ± 0.00" coverage gap in the w/o SWD ablation (gap zero because both groups are fully covered).
  - The reported set size 1.44 and singleton rate 55.9% imply the *implementation* differs from the theorem as stated. Reconcile them and report per-group set sizes.
- **[M] Mislabeled and misworded guarantee.**
  - "Monotonic" Laplace: no monotonicity is enforced. Noise is independent per grid point. Either add isotonic or cumulative post-processing or drop the word. It appears in the title of Contribution 4 and in the Highlights.
  - The guarantee is only a *lower* bound on coverage per group, so it does not guarantee "equity". Over-conservatism for the small group widens the gap.
- **[M] Empirical-Bayes shrinkage** `λ_a = n_a/(n_a+20)`:
  - It is a heuristic with a fixed constant, not Empirical Bayes.
  - It destroys the group-conditional guarantee, contradicting "preserving finite-sample coverage validity".
  - It uses `q̂_marg` computed from all calibration records, which is an additional release over non-disjoint data. Parallel composition no longer applies.
- **[M]** The DKW plus union-bound argument is arithmetically correct (the Laplace tail computation checks out), but it is very loose for small n. Compare with exact order-statistics/Beta-based training-conditional bounds and Mondrian conformal.
- **[m]** Coverage is claimed over `𝒟_a` (undefined). Empty sets are possible when `q̂ < 0.5` and are not discussed. Prediction sets require `a` at inference (disparate-treatment concern).
- **[M]** The end-to-end privacy statement for releasing (model, thresholds, attributions) is never stated. Say how the training ε and `ε_conf` compose given the disjoint split.

### 3.6 DCDA
- **[m]** "Exact completeness" holds only for the continuous integral. A 50-node Gauss-Legendre rule gives an approximation. Report the completeness residual.
- **[M]** Demographic-specific median baselines:
  - They are computed from private data and so are unaccounted for in privacy.
  - They make attributions depend on the protected attribute.
  - They do *not* fix off-manifold interpolation for binary or categorical features (sex, fbs, exang, thal). Straight-line paths still pass through non-physical values.
- **[m]** "Information-theoretic" is rhetoric. IG on `p̂₁` and on MI are standard IG applied to two scalars.
- **[M]** No sanity checks (Adebayo et al.) or ground-truth recovery. Ground-truth recovery is easy to test on the simulated cohort, where coefficients are known. There is no evidence that Channel B tracks epistemic uncertainty.
- **[M]** EDL's vacuity is known to be an unreliable epistemic signal (Bengs et al., cited but not addressed). There is no OOD experiment validating `u` or MI.

### 3.7 Fairness theory
- **[M]** Chouldechova and Kleinberg et al. prove incompatibility of *calibration/predictive parity* with equalized odds, not Statistical Parity versus Equalized Odds. The latter is in Barocas–Hardt–Narayanan. Fix the attribution.
- **[M]** Class-conditional representation alignment for Equalized Odds is not new (e.g., LAFTR-EO, conditional adversarial debiasing). Position the novelty precisely.

---

## 4. Clinical & Biomedical Accuracy

**Statlog Heart**
- **[M]** D=12 but only 11 variables are listed. Chest-pain type is omitted from the text. Also state whether sex is an input feature.
- **[m]** "1988 Cleveland cohort" is imprecise. The data were collected at Cleveland Clinic in the early 1980s and donated to UCI in 1988.
- **[M]** Women in an angiography-referral cohort are a selected population. AHA registry priors do not apply.

**Pima**
- **[M]** The cohort is women aged ≥21 from one Native American community. Generalizing to "metabolic screening" and NHANES priors is unjustified.
- **[M]** Zero-coded missing values (glucose, BP, skinfold, insulin, BMI) are never described. Age is both the protected attribute and an input feature.

**Simulated EHR cohort** (the biggest clinical-realism problem)
- **[C]** The simulation has no demographic effect, so there is nothing to debias. SP/EO ≈ 0 for *every* method, including ERM. Cov. gap is 0.009 for every private model.
- **[C] Degenerate classifier.** At prevalence 12.5%, threshold 0.5 and AUC ≈ 0.67, the models almost surely predict all-negative, making TPR=FPR=0 in both groups. `Δ_EO < 0.001` is therefore a trivial artefact. Using it in the regulatory section ("Δ_EO < 0.001 on acute inpatient cohorts") is misleading. Report predicted-positive rates, sensitivity at fixed specificity, and AUC gaps.
- **[M]** AUC 0.67 from lactate, MAP, SOFA, and similar features is far below published early-warning performance (NEWS2, eCART, and the like reach ≈0.75–0.85). The simulated signal is weak or the model is poor. Include clinical-score baselines (NEWS2, qSOFA, SOFA).
- **[M] Variable realism.**
  - SOFA is a composite of MAP, platelets, creatinine, bilirubin, and P/F ratio, which are also listed as features. This is redundant and collinear. SBP and MAP are also collinear, as are BUN and creatinine.
  - Arterial pH and PaO₂/FiO₂ require arterial blood gases, available mostly *after* a patient is already critically ill. They are informatively missing on wards and are inappropriate for predicting ICU transfer.
  - There is no age, GCS, urine output, vasopressor or ventilation status, comorbidity, or timestamp.
  - The prediction horizon and index time are unspecified.
  - The composite outcome (ICU transfer or 30-day mortality) mixes events with different clinical meaning.
- **[M]** "Parameterized from MIMIC-IV and eICU" is not documented. Release the generator, with correlations and missingness. State PhysioNet DUA compliance.
- **[M]** Channel B discussion refers to "missing laboratory panels", but the model takes complete continuous vectors and no missingness mechanism is described. That narrative is unsupported by any experiment. The XAI section also describes EHR features (lactate, MAP) but supports them with infidelity numbers from Heart and Pima.
- **[m]** The clinical thresholds quoted (lactate >4 mmol/L, MAP <65 mmHg, SBP <90 mmHg) are standard and acceptable.

**Workflow and regulatory**
- **[M]** A singleton `{0}` with 90% coverage under *marginal* conformal is a "no disease" reassurance with up to 10% misses, and these can concentrate in the minority class. Clinically, label-conditional (Mondrian group × class) coverage is needed, especially at 12.5% prevalence.
- **[M]** The abstain-rate implied by the results (44% non-singleton on Heart) is not operationally viable. The `u<0.20` threshold is arbitrary and unvalidated. There is no decision-curve analysis, net benefit, or calibration plots.
- **[M]** The regulatory section overclaims.
  - "Bounding demographic error disparity" is an empirical estimate on ≈7 female positives per fold, not a bound.
  - DP does not "manage retraining risk" (GMLP #10). Budgets accumulate across retraining rounds, which undermines the PCCP claim.
  - Applicability dates and current status of the EU AI Act high-risk obligations and HHS §92.210 should be verified against today's date.
  - Consider moving this section to a short discussion paragraph.

---

## 5. Statistical & Experimental Soundness

1. **[C] The empirical record does not support the headline claims.**
   - On Heart, TRACER-Med is best on only ΔSP and ΔEO. It has the lowest AUC among all private models and Standard EDL. Its Cov. gap (0.045) is worse than DP-SGD (0.001), DP-EDL (0.035), and Fair-ERM (0.032).
   - On Pima, it is *worse* than DP-EDL on ΔSP (0.361 vs 0.308) and ΔEO (0.258 vs 0.203), and worse than non-private Fair-ERM on ΔEO. Its Cov. gap (0.058) is ~10× that of the DP baselines.
   - The abstract reports the favorable Heart/EHR gaps and omits Pima.
   - "Superior calibration" is not supported. Fair-ERM (0.099) and EDL (0.098) have lower Pima ECE.
2. **[C] The premise (DP widens disparities and degrades calibration) is not shown in the paper's own data.**
   - Pima ΔEO: ERM 0.285 → DP-SGD 0.216; EDL 0.345 → DP-EDL 0.203. Heart ΔEO: EDL 0.215 → DP-EDL 0.142.
   - ECE improves under DP on Heart and worsens on Pima.
   - DP behaves like a regularizer. DP-SGD beating ERM on AUC (0.883 vs 0.860; 0.815 vs 0.792) suggests untuned, overfit non-private baselines (20 epochs, no regularization, no early stopping).
   - Run an ε sweep (e.g., ε ∈ {0.5, 1, 2, 4, 8, ∞}).
3. **[C] Noise floor equals effect size.**
   - N=270, 3 folds, ≈29 females per test fold, ≈7 positives among them. Std is often as large as the mean (Cov. gap 0.045±0.05; ΔEO of ERM 0.148±0.08).
   - The ablation row "w/o DP-GCCP" is described as inference-time only, so point metrics should equal the full model. They differ (AUC 0.890 vs 0.863, SP 0.274 vs 0.232, ECE 0.097 vs 0.077). Seed-to-seed variability alone is thus as large as the claimed component effects.
   - "w/o SC-DP" has ΔEO 0.177, identical to Standard DP-SGD's 0.177. Verify this is not a copy or coincidence.
   - Use repeated stratified CV (e.g., 5×10 or more), bootstrap CIs, paired tests, and effect sizes. Define "std" and whether it is over folds or seeds. The 3-fold/3-seed design is ambiguous, and the captions say "3 seeds".
4. **[M] The ablation undermines the conclusions.**
   - Every ablated variant has higher AUC than the full model, and "w/o SC-DP" also has lower ΔSP and ECE.
   - w/o DP-GCCP shows coverage disparity 0.046 vs 0.045, so there is no benefit. The text still says DP-GCCP "strictly guarantees … preventing minority undercoverage".
   - Missing ablations:
     - κ=0 (prevalence-only clipping);
     - γ=0;
     - marginal vs class-conditional SWD;
     - the evidential head on its own.
   - Without these there is no evidence for the *Fisher* contribution.
   - The ablation is on Heart only, though the text says "across the benchmarks".
5. **[M] Missing baselines.**
   - Cited but not run: DPSGD-F (Xu), Tran et al., and Esipova et al.
   - Missing: Hardt post-processing thresholds (these achieve ΔEO≈0 cheaply), group-wise Mondrian conformal, non-DP-fair-EDL, logistic regression / GBDT / DP-LR (Pima AUC 0.826 is about logistic-regression level), and clinical scores for EHR.
6. **[M] Metrics.**
   - ΔEO is defined as the mean of |ΔFPR| and |ΔTPR|, while the usual definition is the max. State this.
   - A fixed 0.5 threshold is clinically arbitrary. Add threshold-free metrics (Subgroup AUC gap, listed in §5.3 but never reported), plus PR-AUC, Average Set Cardinality, marginal coverage, and Faithfulness Rank Correlation. All were announced in the metric suite and none appear in the tables.
   - ECE with 10 bins on ≈90 test samples is unreliable.
   - Infidelity is scale-dependent and not comparable across softmax vs evidential models or `p̂` vs MI. Channel B infidelity is never reported.
7. **[M] The MIA audit does not support the abstract.**
   - The text says private models reach "0.500–0.547". Table 1 gives Standard DP-SGD on Heart = 0.571, equal to non-private Standard EDL (0.574).
   - MIA AUC below 0.5 (0.498) is bolded as "best". The EHR non-private models show no leakage (0.510–0.514), so that audit has no power.
   - With ε≈15, near-chance MIA cannot "corroborate" the formal bound. Use stronger attacks (LiRA) and report TPR at low FPR. Report per-group MIA, since disparate vulnerability is directly relevant to the paper's thesis.
8. **[M] The conformal baselines are undefined.** Which conformal procedure is applied to ERM, EDL, and the other baselines? Report per-group set size, singleton rate, and class-conditional coverage. Also report coverage when the theorem's τ_a would be vacuous.
9. **[M] Reproducibility.**
   - Unreported: κ, γ, S̄_a, p_a values used, λ_fair, s_a, τ̄_y, M (projections), learning-rate/clip tuning, the accountant, the train/cal split sizes, preprocessing, and how hyperparameters were chosen.
   - Selecting hyperparameters on the same folds as reporting is selection bias, and tuning also consumes privacy budget.
   - "3-layer (64, 32)" is ambiguous.
10. **[M] The claim that benefits are "most prominent on imbalanced cohorts" is untested.** Two datasets differ in many ways, and the EHR cohort is 30% minority with no effect. Do a controlled minority-fraction sweep by subsampling.
11. **[M] Figures.**
    - Fig. 1 plots ΔSP, though the paper argues EO is the primary criterion.
    - Fig. 2 uses only the EHR cohort, the one with no disparity.
    - The Fig. 4 radar uses incommensurate axes. `1−MIA AUC` is capped near 0.5, and `1−Cov.Gap` and `1−ΔEO` compress near 1. The claim that TRACER-Med "encloses" the baselines is contradicted by the table (e.g., DP-EDL's AUC and ECE on Heart).
    - Add error bars, Pareto fronts, and calibration/reliability plots.

---

## 6. Minor Issues, Writing, LaTeX, Micro-Edits

**Structure and consistency**
- §4 opener says "four co-designed modules", but five are listed. The Introduction lists four pillars while the Trilemma has three tensions. "Trilemma" implies an impossibility result, and none is given. Contribution 1 ("analytically articulate … showing why") contains no analysis.
- §5.2 says "five representative paradigms" and then lists six (including the proposed method).
- §5.3 lists six metric dimensions, but Table 1 has eight columns with different content.
- Ablation text says "across the benchmarks", but Table 2 is Heart only. The set size (1.44) and singleton rate (55.9%) for marginal CP appear in the text, but the DP-GCCP counterparts are not given.
- "Ablation w/o SWD" uses a different name from "CC-SWD" in the method section.
- The Table 1 caption says bold is the best among private models, but TRACER is bold in few cells. Ties are bolded inconsistently (EHR ΔSP/ΔEO "<0.001" is not bolded for anyone). Mean/std precision is inconsistent ("0.083 ± 0.02", "0.001 ± 0.001", "<0.001", "0.000 ± 0.00"). State whether std uses ddof 0 or 1.
- Algorithm 1: add κ, γ, λ_fair, M and the accountant to the inputs. The "epoch" loop is inconsistent with Poisson steps (T = E/q). `|B|` should be `qN`.

**Notation collisions**
- `τ`: decision threshold, `τ_a` (conformal buffer), `τ̄_y` (target rate).
- `q`: sampling rate vs `q̂_a` (conformal quantile).
- `s_i` (non-conformity) vs `s_a` (calibration weight).
- `𝒮`: output set in the DP definition vs `𝒮_a` (calibration partition) vs `S` (Dirichlet strength).
- `ξ`: Gaussian vs Laplace noise.
- `e`: epoch vs evidence vs Euler's constant. `t`: path parameter, grid point, step index. `E`: epochs vs expectation. `M`: three uses.
- `a_lrelu` vs group `a`. `λ_RDP`, `λ_fair`, and `λ_a` all appear.
- `θ`: parameters vs projections. `C` and `𝒞` (clip vs set). `Δ`: sensitivity vs disparities.
- `β_e = min(1, e/10)` is stated three times; state it once.

**LaTeX**
- `\theoremstyle{plain}` is applied to `definition`. Use `\theoremstyle{definition}`. `proposition` and `corollary` are unused. Check for `amsthm`/`elsarticle` proof-environment warnings.
- `\resizebox{\textwidth}{!}` on the 9-column Table 1 under 12 pt shrinks the font badly. Use `\footnotesize` with `tabular*` or split the table. `table*`/`figure*` have no effect under single-column `3p`.
- Use `\theoremstyle` and `\operatorname` consistently. `\text{Range}` should be `\operatorname{Range}`.
- `Dvoretzky-Kiefer-Wolfowitz` should use en-dashes. `non-conformity` vs `nonconformity`: pick one.
- Equation (KL) line breaks are fine. Verify `elsarticle-num` renders the `\affiliation` fields (add postcode if the journal requires it).
- The abstract (~450+ words) likely exceeds the journal's limit (check, ~250). Highlights are within 85 characters.
- The title is ~25 words. Shorten it.

**Abbreviations**
- Undefined at first use in the abstract: DP-EDL, EHR, PAC, MIA, SWD. Also undefined: MDR/IVDR and SOFA. OOD is defined twice (Introduction and Related Work). EDL is defined only in Related Work.

**Phrasing and tone**
- Overclaiming words should be toned down: "substantially mitigates", "strictly guarantees", "exact", "successful harmonization", "mathematically principled, auditable", "establishes". The conclusion's claim of "faithful dual-channel explanations" is unsupported.
- Remove ornamental language ("sharp structural friction", "premier international regulatory guidelines", "acute Trilemma").
- "As proven by Barocas et al., Chouldechova, and Kleinberg": see §3.7.
- Notation slip: `p̂₁(x) = P(Ŷ=1|x)` should be `P(Y=1|x)` under the model.

**References** (verify against the bib; I could not open it)
- References are largely 2016–2022. The commit log mentions 2025–2026 additions, but the text cites almost nothing newer than 2024. Add 2023–2026 work on DP-fairness (e.g., Mangold et al., Berrada et al., Jagielski et al., Cummings et al.), private conformal prediction, fair conformal prediction in medicine, EDL critiques, and DP in medical AI.
- `tibshirani2019conformal` (covariate shift) and `romano2019conformalized` (CQR) are cited as generic group-conditional conformal references. The proper reference is Mondrian conformal (Vovk), which is missing.
- `patel2022towards` is cited both for explainability (Introduction) and privacy auditing (Related Work). Check which is correct.
- Confirm the year for Esipova et al.

**Declarations**
- The generative-AI declaration names "Claude 3.5 Sonnet". Elsevier requires an accurate disclosure, so verify the tool and version. This conflicts with your global instruction to keep Claude identity out of the project, and it is your call to resolve. Journal policy takes priority for the manuscript text.
- The Ethics statement says the EHR cohort is "derived from" MIMIC-IV/eICU. Add data-use-agreement compliance.

---

## 7. Final Verdict & Checklist of Actionable Revisions

**Verdict: Major revision (borderline reject).** The framing is attractive, but three formal results are incorrect or unproven as stated (conformal privacy, subsampled-Gaussian RDP, "Fisher" derivation). The method as trained differs from the method as described (CC-SWD). The evidence is weak and partly contradicts the claims.

**Must fix (blocking)**
1. Correct the DP-GCCP privacy proof. L1 sensitivity is up to M. Use a histogram/tree mechanism or scale the noise to M/ε. Treat `n_a` as public or noisy. Recompute τ_a and re-run.
2. Reconcile Theorem 2 with the implementation. At the used n_a, the buffer exceeds 1 and sets become trivial. Report per-group set sizes. Drop or properly justify "monotonic", "PAC equity", and the shrinkage guarantee. Account for the extra release in `q̂_marg`.
3. Replace the RDP bound with an exact accountant (σ=0.8, q=0.18 is outside the stated regime). Name the tool. Remove "exact". Use the expected batch size `qN` in the Adam step.
4. Fix the gradient formula (trigamma, sigmoid derivative). Either derive a real Fisher-based rule or rename the method. Resolve the sign of the S̄ exponent. Define F-SNR or drop the claim. Document the source of `p_a` and `S̄_a`.
5. Make CC-SWD actually what is described. A per-sample surrogate must be named as such, with `τ̄_y` and `s_a` defined and shown to be data-independent. If not, account for the privacy cost.
6. Redesign the EHR simulation: add group-dependent shifts and realistic variables, a defined prediction horizon, and a non-degenerate operating point. Drop the "Δ_EO < 0.001" claims.
7. Rewrite the abstract and conclusions to match the table. Report Pima's worse fairness and coverage, and the MIA range including DP-SGD's 0.571. Remove "corroborated by MIA".

**Should fix (major)**
8. Run repeated CV with CIs and paired tests. Sweep ε (including ∞) and the minority fraction. Use tuned baselines and report tuning protocols.
9. Add baselines: DPSGD-F, Tran et al., Hardt post-processing, marginal-SWD vs class-conditional-SWD, Mondrian conformal, LR/GBDT/DP-LR, clinical scores.
10. Add ablations for κ=0, γ=0, and each component on all datasets. Explain why ablated variants beat the full model on AUC and why the "inference-only" ablation changes point metrics. Check the 0.177 coincidence.
11. Report every announced metric (PR-AUC, subgroup AUC gap, set size, singleton rate, faithfulness correlation, Channel B infidelity) and clinical utility (decision curves, class-conditional coverage).
12. Validate DCDA: sanity checks, ground-truth recovery on the simulation, DP-safe or public baselines, and OOD evidence for `u` and MI.
13. List all hyperparameters (κ, γ, λ_fair, s_a, τ̄_y, M, accountant, split sizes, preprocessing) and fix the Heart feature list (D=12, chest pain type).
14. Redo the radar figure (or drop it), plot ΔEO in Fig. 1, and use Heart in Fig. 2.

**Polish**
15. Fix the module and baseline counts, notation collisions, abbreviations, `definition` style, table sizing, abstract length, title length, and the Chouldechova/Kleinberg misattribution.
16. Update references through 2026, add Mondrian conformal and the fair-private literature, and verify the questionable citations.
17. Tone down the regulatory section and verify its dates.
18. Tone down the overclaiming language throughout.

I can turn this into a point-by-point revision plan against `manuscript/main.tex` or the source files in `src/`. I can also start with the highest-value fixes: the conformal privacy proof, the RDP accountant, and the ablation re-run.
