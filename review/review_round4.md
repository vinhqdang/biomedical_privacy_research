# Referee Report: *TRACER-Med*

I reviewed the LaTeX source as submitted. I did not compile it, run the code, or check the `.bib` file. Comments flagged "verify" are items I could not confirm from the text alone.

---

## 1. Overall Assessment & Summary of Contributions

The manuscript proposes a four-module pipeline for tabular clinical classification:

1. An evidential Dirichlet (EDL) backbone.
2. SC-DP-SGD, which uses group-dependent clipping bounds with noise calibrated to the largest bound, C_max.
3. A DP group-conditional conformal layer (DP-GCCP).
4. A two-channel Integrated-Gradients-style attribution (DCDA).

A Sliced-Wasserstein (SWD) fairness regulariser sits outside the DP loop. It is evaluated on Statlog Heart (N=270), Pima (N=768), and a 10,000-patient synthetic EHR cohort.

The topic is timely and the ambition is appropriate for *AIM*. The privacy proof is honest about adjacency and about decoupling SWD from the private step. The manuscript also admits several limitations.

In its current form, however, the evidence does not support the headline claims, for these reasons:

- **Privacy and conformal proofs.** The conformal proposition proof does not establish its claim. The RDP bound in the privacy proof is an asymptotic heuristic that is invalid at the σ used.
- **Empirical support.** The empirical support is one small dataset with 3 runs, and its test groups hold single-digit numbers of positive cases. Several results are internally impossible or contradictory.
- **Missing comparisons.** The closest prior work, group-specific clipping in DPSGD-F (Xu et al.), is cited but neither discussed as a competitor nor compared against.
- **Overclaiming.** The regulatory and clinical sections claim compliance and workflow readiness that the experiments cannot support.

Recommendation: **major revision**. As it stands I would not send it to external review without the Critical items below addressed.

---

## 2. Major Strengths

- The paper integrates four trustworthiness pillars and states the privacy threat model explicitly: add/remove adjacency, Poisson subsampling, and disjoint calibration data.
- It separates SWD from the DP step instead of silently breaking per-sample sensitivity.
- SC-DP-SGD is simple and cheap. The weight normalisation Σ p_a w̃_a² = 1 is correct. The Algorithm 1 formula (p_a^{-γ}/√Σ p_b^{1−2γ}) is consistent with the main-text formula, since the factor 2^{-γ} cancels.
- The percentage reductions in the text are arithmetically correct: 42%, 27%, 53% and 113% all check out.
- The authors acknowledge synthetic-data, binary-attribute and longitudinal limitations.
- Code and OpenML IDs (53 and 37) are given, which is good for reproducibility.
- Reporting negative or neutral results, such as Pima where SC-DP ≈ uniform clipping, is commendable. The interpretation of those results, however, needs repair (see Section 5).

---

## 3. Critical Methodological & Mathematical Issues

### 3.1 Privacy (Theorem 1)

**C1. The per-step RDP bound is not valid at σ = 0.8.**
- The proof asserts that the Poisson-subsampled Gaussian has RDP "at most q²λ/(2σ²)".
- That expression is only the small-q, large-σ, low-order approximation. At order 2 the exact value is ln(1 + q²(e^{1/σ²} − 1)).
- At σ = 0.8, e^{1/σ²} − 1 ≈ 3.8, versus 1/σ² ≈ 1.56 in the approximation. The approximation therefore understates the exact order-2 RDP by a factor of about 2.4.
- The theorem's reported ε values come from "the exact numerical accountant", not from this bound, so the proof and the numbers are inconsistent.
- Fix: state ε(λ) via the exact Mironov–Talwar–Zhang expression for integer λ (or cite the accountant), and drop the closed form.

**C2. Essential accounting inputs are missing.**
- T (steps or epochs), learning rate, optimiser and architecture are not reported, so ε = 15.36 / 6.13 / 2.49 cannot be reproduced.
- The accountant needs the actual training-set size. With stratified 3-fold CV and a carved-out calibration set, N is not 270 (it is about 180 or fewer). Please state the q actually used.
- Fixed-size shuffled mini-batches of 32 do not enjoy Poisson amplification. If the implementation does not Poisson-sample, the accountant is invalid.

**C3. "Fixed population priors" are not independent of the private sample.**
- The priors used (32/68, 46/54, 29/71) are exactly the empirical group frequencies of the datasets. They are not census or registry values.
- If w̃_a is computed from sample counts, the "no auxiliary leakage" statement in Section 3.1 is false.
- Group sizes |S_a| also drive the conformal noise scale and the DKW buffer (see C6).

**C4. The overall pipeline privacy is not accounted for.**
- The following are all computed from private data without DP accounting:
  - median imputation (Pima)
  - feature scaling
  - the "median healthy profile" baseline x⁰ in DCDA
  - hyperparameter selection
  - the SWD "pre-training" (see C9)
- The abstract reports only ε_model. ε_conf is never specified anywhere in the experiments.
- ε = 15.36 is a weak guarantee. Describing it as "rigorous privacy accounting" in the abstract is misleading. A fixed target ε sweep (for example 1, 3, 8) with σ calibrated to it is the expected practice, rather than one σ yielding incomparable ε across datasets.

**C5. The "bounding sensitivity by C_max" novelty is trivial, and the cost is unanalysed.**
- The clipping norm of group a is C_a, and the sensitivity is max C_a. This is a restatement of the definition, not a result.
- Numerically (γ = 0.5, Heart): w̃_female ≈ 1.25 and w̃_male ≈ 0.86, so C_max = 1.25C.
- Noise is σ·1.25C, but male gradients are clipped at 0.86C. The majority group's effective noise-to-signal ratio rises to ≈ 1.17σ.
- No analysis or γ sweep is given, and γ = 0.5 is hard-coded.
- The premise (Gap 1) that minority gradients are clipped more is never verified, for example with per-group clipping rates or gradient-norm statistics.
- The theorem does not prove the 27%/42% disparity reduction, yet Contribution 3 says "we prove… while reducing demographic disparity by 27%".

### 3.2 Conformal Prediction (Proposition 1 and Eq. 20)

**C6. The proof does not establish the claim.**
- Define q̂_a through the level 1 − α + τ_a + ξ_a. DKW gives F_a(q̂_a) ≥ F_S(q̂_a) − τ_a ≥ 1 − α + ξ_a.
- When ξ_a < 0 the target falls below 1 − α. The proof bounds ξ_a ≥ −ln(4/β)/(|S_a|ε), which does not cancel.
- The Laplace tail must be added to τ_a, so the buffer should be τ_a + b·ln(…).
- Separately, the one-sided tail is P(ξ < −x) = ½e^{−x/b}. The stated bound is loose, but that is harmless.

**C7. The mechanism itself is unproven and data-dependent.**
- "Perturb the level by a scalar Laplace, then invert the empirical CDF" is not a standard DP primitive. It needs its own proof, for example via the monotone ℓ∞ stability of F.
- The noise scale 1/(|S_a|ε) depends on |S_a|, which changes under add/remove adjacency and is itself private.

**C8. Vacuous sets in the experiments.**
- The DKW buffer is far more conservative than the exact split-conformal quantile ⌈(n+1)(1−α)⌉/n.
- For Heart, the minority calibration subset is roughly 10–20 patients, so n = 15 gives τ = √(ln 40 / 30) ≈ 0.35. The level is then 1.25 > 1, and the set is {0,1} always.
- This matches the reported evidence: "average set size 1.84 ± 0.13" for a two-class problem, and "97.1% marginal coverage" at a 90% target.
- Coverage is achieved vacuously. The abstract does not even say which dataset that set size refers to.
- The "coverage gap 0.000 ± 0.00" in the w/o-SWD ablation is the signature of trivial sets.
- The Section 7.2 workflow ("act only on singleton sets") would abstain on about 84% of patients.
- Group-conditional coverage is also stated as a PAC-type "w.p. ≥ 1−β" guarantee, which is inconsistent with the classical exact marginal guarantee the introduction invokes. The exchangeability assumption is never stated.

**C9. The conformal gain is not evidenced and is contradicted by the tables.**
- Heart gap: DP-GCCP 0.045 ± 0.05 versus marginal CP (w/o DP-GCCP) 0.046 ± 0.03. This is no difference.
- Standard DP-SGD achieves 0.001 and DP-EDL 0.035, both better than TRACER-Med's 0.045.
- On Pima, TRACER-Med has gap 0.058 versus 0.006/0.007 for DP-SGD/DP-EDL, which is 8–10× worse.
- On the EHR cohort, all models have an identical gap of 0.009.
- Section 6.3's claim that marginal CP "frequently incurs substantial subgroup coverage deficits" is therefore unsupported.
- It is unclear whether baselines used marginal CP, group-conditional CP, or private or non-private quantiles.

### 3.3 Evidential Deep Learning

**C10. Aleatoric versus epistemic is mis-stated.**
- The manuscript says EDL "analytically separates aleatoric entropy from epistemic vacuity". It then defines H(p̂) as total uncertainty.
- The standard decomposition is total = H(E[p]), aleatoric = E[H(p)], and epistemic = their difference (mutual information). Both have closed forms in the Dirichlet case, via digamma functions.
- Using u = K/S as "epistemic" is a Subjective-Logic convention, not a separation of aleatoric uncertainty.
- The authors cite Bengs et al. and Charpentier et al. (EDL's epistemic estimates are not faithful) but then still claim "calibrated uncertainty". No OOD or error-detection evidence is shown (AUROC, risk–coverage).

**C11. The loss is under-specified.**
- λ_reg, the KL annealing schedule (required in Sensoy et al.), and the per-sample ℓ_i in the clipping step are all missing. L_EDL is written as a batch sum, whereas clipping needs the per-sample loss.
- The notation z is overloaded (see Section 6).

### 3.4 Sliced Wasserstein Alignment

**C12. The DP guarantee is void if SWD uses private data.**
- The manuscript says SWD is applied "on public/unlabelled reference data". No such dataset exists for the 270-patient Heart cohort or for the synthetic EHR cohort.
- The ablation shows SWD matters (w/o SWD: ΔSP 0.259 versus 0.232), so it is used in the reported runs.
- Please state exactly which data trained the encoder. If it was the training split, the stated privacy guarantee does not cover the released encoder.

**C13. SWD aligns the wrong thing.**
- SWD matches P(z|A=0) to P(z|A=1) unconditionally on y. That enforces demographic-parity-like independence.
- The paper itself argues that SP is inappropriate under unequal base rates and that ΔEO is the primary criterion. EO-consistent alignment should be class-conditional, P(z|A,Y=y).

**C14. Citation and sampling.**
- Kolouri 2019 is the *generalised* SW paper. The classical SWD references (Bonneel/Rabin) are missing.
- The number of projections M and the loss weight are not given.

### 3.5 Attribution (DCDA)

**C15. Channel A does not measure support for the diagnosis.**
- ∂ ln α_ŷ ignores the evidence for other classes. A feature that increases evidence for both classes increases α_ŷ and is credited as "supporting" ŷ, even though p̂_ŷ may not change.
- A logit-margin or ln p̂_ŷ attribution is more defensible.

**C16. The two channels are not independent.**
- u = K/S with S = Σα, so Channel B is essentially the negative attribution of total evidence, and Channel A is dominated by the same quantity.
- No correlation analysis or sanity checks are given (Adebayo et al. randomisation tests, deletion/insertion).

**C17. The x⁰ baseline leaks.**
- "Median profile of healthy individuals" is computed from private patients without DP. It is also "demographic-specific" and so depends on A.
- On Heart, interpolating along categorical features such as thal and chest pain type produces non-physical points.
- "Closed-form structure" is overclaimed: IG still needs numerical quadrature, and the number of steps is not given.
- Dual-channel attribution has prior work (Antorán et al. CLUE, Perez et al., and information-theoretic Shapley attributions of predictive uncertainty by Watson et al.), so novelty is narrower than claimed.

### 3.6 Fairness Definitions

**C18. Impossibility theorems are misattributed.**
- Chouldechova and Kleinberg et al. concern calibration-within-groups versus error-rate balance. The incompatibility of independence (SP) and separation (EO) under unequal base rates is from Barocas–Hardt–Narayanan, or follows from a short direct argument.
- "EO is primary to avoid miscalibrating clinical risk" is wrong in direction. EO and within-group calibration are exactly the pair that conflicts (Kleinberg).
- No per-group ECE is reported, although "subgroup miscalibration" is Gap 2.

**C19. The metrics are underspecified.**
- The decision threshold is not stated. Fairness metrics are computed at a fixed threshold only, and there is no threshold-free analysis.
- The metrics are defined for binary Ŷ, while the framework is K-class.

---

## 4. Clinical & Biomedical Accuracy

1. **Statlog Heart.**
   - It is a 270-patient, late-1980s Cleveland angiography-referral sample, not a "real-world clinical diagnostic cohort".
   - Features such as `ca` (vessels by fluoroscopy) and `thal` (thallium scan) are downstream diagnostic findings, close to target leakage for a "diagnosis" model.
   - Sex differences in CAD base rate are real, so reducing ΔSP by forcing parity is not clinically desirable.
   - Text says 13 variables while the tables say D = 12. If sex was excluded as the protected attribute, say so. The EHR cohort includes the group indicator as a feature, which is inconsistent treatment.

2. **Pima.**
   - The "age ≤ 30 = 46%" figure looks inverted: the median age in Pima is 29, so at least about half are ≤ 30. Please verify.
   - Insulin is ~48% zeros and skin thickness ~30%. Median imputation of that scale, done before the split, is leakage and also a DP violation.
   - Age is a primary clinical risk factor, so equalising predictions across age is of doubtful clinical meaning.
   - "Onset of diabetes" is misleading; the label is WHO-criteria diagnosis within five years.

3. **Synthetic EHR cohort.**
   - The data-generating process is undescribed: correlation structure, outcome model, and whether the minority group differs at all.
   - SOFA (and Charlson) are included alongside their component labs (creatinine, platelets), which makes them redundant. SOFA's other inputs (GCS, PaO₂/FiO₂, bilirubin, vasopressors, urine output) are absent.
   - "30-day acute clinical deterioration / ICU mortality" conflates three different endpoints.
   - AUC of 0.67 is far below real ICU-mortality performance (about 0.8–0.9), so the simulator has very weak signal.
   - Calling it "standardized clinical inpatient benchmark" contradicts "synthetic".
   - Highlight 5 and the Regulatory section present it as an evaluation on an "acute inpatient cohort".
   - **Fairness numbers here are trivial.** Δ_SP and Δ_EO < 0.001 for every model, including non-private ERM. With a low-prevalence outcome (Brier ≈ 0.077) and AUC 0.67, thresholded predictions are almost all negative, so TPR ≈ FPR ≈ 0 in both groups. Citing "Δ_EO < 0.001 on acute inpatient cohorts" in Section 7 as evidence of compliance is not valid.

4. **Qualitative attribution findings (Section 6.5).**
   - Lactate > 4 mmol/L, SBP < 85, and SOFA are recovered because they were put into the simulator. The result is circular and is not clinical validation.
   - The text quotes infidelity "on Heart/Pima" but discusses lactate and SBP, which exist in neither dataset.
   - SBP < 85 is not a standard threshold (< 90, MAP < 65, or qSOFA ≤ 100 are).
   - "Channel B reflects missing or borderline indicators" is unsupported. The data has no missingness, and EDL vacuity reflects distance from training evidence, not missing tests.
   - Fig. 3 shows only "illustrative" cases, with no baselines or clinician assessment.

5. **Workflow (Section 7.2).**
   - The u < 0.20 threshold (S > 10 for K=2) is arbitrary and uncalibrated.
   - No abstention rate, selective-accuracy (risk–coverage) curve, decision-curve/net-benefit analysis, or human-factors evaluation is given.
   - With set size 1.84, the proposed system would abstain on most cases.
   - "Use A at inference to pick the threshold" is compared to sex-stratified lab reference ranges. That is a weak analogy, and it raises disparate-treatment concerns (Section 1557). It also conflicts with EO-style fairness for a model that does not use A.

6. **Regulatory claims.**
   - GMLP Principle 3 concerns representativeness of the *data*; clipping weights do not make a training set representative.
   - GMLP Principle 10 (monitoring and retraining risk) is not "satisfied via DP".
   - DP does not equal HIPAA compliance and is not a "legally defensible" guardrail for PCCPs.
   - Under the EU AI Act, medical devices are high-risk via Art. 6(1)/Annex I (MDR/IVDR), with obligations phased in. Verify current application dates, since the timeline has been under revision.
   - Section 1557 (§92.210) applies to covered entities, not model developers, and enforcement posture has shifted. Please verify the current status.
   - The FDA PCCP guidance title is "AI-Enabled Device Software Functions", not "AI/ML-enabled".
   - "TRACER-Med satisfies these standards" should become "is designed to support documentation of…". A single ΔEO point estimate on 270 patients is not an Article 10/13/14/15 compliance demonstration.

---

## 5. Statistical & Experimental Soundness

**S1. Sample size and variance.**
- Heart test folds have about 90 patients, with ~29 women and ~8 positive female cases. ΔEO differences of 0.06 are about one patient.
- n = 3 for all mean ± SD. "3-fold CV with seeds 42–44" is ambiguous: 3 runs or 9? Is the SD across folds or seeds? Stratified by what (label, or label × group)?
- There are no confidence intervals, no paired tests, and no multiple-comparison control.
- Bolding uses means only and ignores overlapping SDs (for example ECE 0.074 vs 0.077).
- Standard deviations are inconsistent in precision (±0.001 beside means like 0.999).

**S2. Impossible ablation values.**
- DP-GCCP is post-hoc conformal. It cannot change AUC, ECE, ΔSP or ΔEO of the point predictor.
- Yet Table 3 shows w/o DP-GCCP: AUC 0.890 vs 0.863, ECE 0.097 vs 0.077, ΔSP 0.274 vs 0.232. Either different training runs were used (so the ablation measures run-to-run noise, comparable to the claimed effects) or there is a bug.
- The interpretation in Section 6.6, item 3, "group-conditional calibration reinforces subgroup equity post-hoc", is therefore invalid.
- w/o SC-DP reports ΔEO 0.177, identical to Standard DP-SGD's 0.177 (copy error?).
- w/o SC-DP is "DP-EDL + SWD". Its ΔEO (0.177) is *worse* than DP-EDL (0.142), which suggests SWD is harmful alone or that variance dominates.
- Every ablation has *higher* AUC than the full model (0.863 vs 0.879–0.890). This is not discussed.
- The text says ablations were run "across the benchmarks", but Table 3 is Heart only.

**S3. Confounded baselines.**
- TRACER-Med gets SWD, while DP-EDL does not.
- The closest competitors are missing: DPSGD-F (Xu et al.), Tran et al., Esipova et al., and per-group-clipping baselines. All are cited in Related Work but not run. DPSGD-F also uses group-specific clipping, so SC-DP-SGD's novelty over it must be argued.
- There are no simple tabular baselines such as DP-logistic regression or gradient boosting. Logistic regression reaches about 0.90 AUC on Statlog and about 0.83 on Pima, above the reported networks (ERM: 0.860 and 0.792).
- There is no non-private TRACER-Med, so the cost of DP versus the cost of the method cannot be separated.
- The baseline-fairness comparison is incomplete. Only a single ε per dataset is evaluated, with no ε sweep and no Pareto fronts, so "mitigation of the privacy–fairness trade-off" is not characterised.
- Private models beat non-private ERM on Heart (AUC 0.883 vs 0.860; ECE 0.146 vs 0.199). This suggests weakly regularised, untuned baselines. The identical σ = 0.8, C = 1.0 is used for all, and tuning is not described.

**S4. Selective reporting.**
- The abstract highlights only Heart. On Pima, TRACER-Med is worse on the declared *primary* criterion: ΔEO 0.258 vs 0.203 (DP-EDL) and 0.216 (DP-SGD). The text mentions only ΔSP and calls a 17% worse result "slightly lower disparity".
- The claim of "superior probability calibration" on Pima is false: ECE 0.108 vs 0.108 (DP-EDL), and 0.098–0.099 for non-private baselines.
- AUC 0.826 ± 0.02 vs 0.812 ± 0.01 is not a clear win.
- On Heart, TRACER-Med has the lowest AUC among EDL/DP models (0.863 vs 0.883–0.890).
- The EHR column offers no discriminating evidence for any claim.
- Contradiction: the introduction says DP degrades attributions, yet Standard DP-SGD has the lowest infidelity. TRACER-Med's 6× improvement over Standard EDL (0.015 vs 0.098) has no mechanism. Infidelity is also not comparable across models with different output scales.
- The claim that DP induces miscalibration is mixed. Heart ECE improves under DP-SGD (0.199 to 0.146), while Pima worsens (0.126 to 0.202).

**S5. Metrics.**
- The following are listed in Section 5.3 but never reported:
  - PR-AUC
  - subgroup AUC gap
  - faithfulness rank correlation
  - per-dataset marginal coverage
  - set size
- ECE with 10 bins on ~90 test points is dominated by noise.

**S6. MIA.**
- A shadow-loss-threshold attack with AUC near 0.5 is a weak audit. Use LiRA (TPR at low FPR) or canary-based auditing.
- Empirical MIA cannot "corroborate" a formal guarantee; it only gives a lower bound on leakage.
- Standard DP-SGD on Heart has MIA 0.571, outside the stated "0.500 to 0.547" range and comparable to non-private EDL (0.574). That shows ε = 15 gives little protection.
- The abstract says "0.50–0.54", but the table's maximum is 0.547, which rounds to 0.55.
- Conformal outputs and attributions are not audited for leakage.

---

## 6. Minor Issues, Writing, LaTeX Syntax, Micro-Edits

### Abstract, highlights and framing
- The abstract is long (about 300+ words). Check the journal's limit.
- "Trilemma" names three tensions, but the manuscript sets up four pillars and then four bullets. Contribution 1 claims an "analytical" formulation, but none is given. The paper contains no trade-off theorem and no dedicated empirical dissection.
- "Rigorous privacy accounting (ε = 15.36)" reads as an overstatement.
- Highlight 2, "bounds sensitivity by C_max", is trivial. Highlight 5 describes a simulated cohort as an "inpatient" evaluation.
- "Adaptive" in the TRACER-Med acronym is not accurate: the weights are fixed priors.
- "Dual-Channel" separates "risk from doubt" is stronger than the evidence.
- The Introduction cites generic AI-vs-specialist reviews for "acute mortality triage". The claim is not supported by those references.
- Intro bullet 4 ("Gradient perturbations degrade feature attributions") has no citation and is contradicted by the results.

### Notation inconsistencies
- `z` is used for pre-softplus logits, for latent representations h_θ(x), and for a record z = (x, y, a). Also `f_θ` versus `h_θ` is used for the network.
- `\hat{Y}` vs `\hat{y}`.
- `\mathcal{B}'` is used in Δ₂ but never defined.
- y_{ik} (one-hot) and ℓ_i are undefined.
- `K`-class definitions are mixed with binary-only fairness metrics.
- `λ_RDP` (order) and `λ_reg` (weight) are fine, but `α` is used for Dirichlet parameters, α_conf, and (in Dirichlet context) concentration. Consider renaming Dirichlet parameters.
- "Cov. Gap", "Cov. Disparity", "Subgroup Coverage Gap" and "coverage gap" are used interchangeably.
- Both `R\'enyi` and `R{\'e}nyi` appear.
- `N` is used for dataset size, `N_cal` for calibration, and the symbol in the theorem is ambiguous with respect to the training split.

### LaTeX
- `\textbf{$\Delta_{\text{SP}}$}` in table headers: `\textbf` does not bold math. Use `\boldsymbol`/`\bm` or drop it.
- `\resizebox{\textwidth}{!}` on a 9-column table in 12pt shrinks text to unreadable size. Split the table (one per dataset) or use `\footnotesize` with `tabular*`/`tabularx`.
- Figure 4's file (`fig4_...`) is cited first in Section 4, so it prints as "Figure 1", while "fig1" is "Figure 2". The radar chart is a results figure sitting in Methods ("evaluated in Figure 1") before results are introduced. Axis normalisation and direction (ECE and gaps are lower-better) are unspecified. Reorder or move it.
- The Fig. 1 caption says "across privacy, utility and parity", but the axes show only AUC and ΔSP. A ΔEO panel is needed, since EO is called primary.
- `algorithmic` is deprecated. Prefer `algpseudocode` (algorithmicx).
- `\affiliation[inst1]{organization=…}` requires the new elsarticle (≥ 3.0) syntax; confirm the journal template version.
- Double blank line before §2.4; "Writing - original draft" in CRediT should use an en-dash.
- A personal e-mail address is given for the corresponding author. Many journals prefer an institutional address.
- Add an ethics statement (public datasets, synthetic cohort) and move the generative-AI declaration to the Elsevier-prescribed heading and wording, naming the tool(s) used and for what.

### Prose
- "Singlet prediction set" should be "singleton".
- "Premier international regulatory guidelines" is awkward. "Directly aligns/instantiates/satisfies" is repeated overclaiming; use "supports".
- "Standardized … synthetic" is contradictory.
- The "Empirical Results and In-Depth Discussion" section contains very little discussion (no failure analysis, no limitations of results).
- "Co-pilot" is jargon.
- The baseline list calls TRACER-Med a baseline bullet ("five baselines and three ablation variants") and nests the ablations inside the itemize. Restructure.
- "Standard ERM" with ECE 0.199 indicates poor training; please explain.
- "Equalized Odds" is defined as an *average* gap and named ΔEO; fine, but some papers use the max or the sum, so cite carefully.

### References
- Newest cited items are from 2024 (`wang2024fair`, `zhang2024differentially`, `charpentier2024evidential`, `fda2024pccp`). For a submission in October 2026, 2025–2026 literature on fair DP-SGD, private conformal prediction, uncertainty attribution, LiRA-based DP auditing, and FDA/EU AI Act updates should be incorporated.
- I could not verify `wang2024fair`, `zhang2024differentially` or `perez2022uncertainty`. Add DOIs and confirm they say what is claimed.
- `romano2019conformalized` is CQR, not group-conditional or Mondrian CP. Cite Vovk's Mondrian CP and Romano et al. 2020 ("With malice toward none").
- Missing biomedical-DP and DP-explanation references, for example Ziller/Kaissis (medical-imaging DP) and Patel et al. (DP explanations); also Shokri et al. (explanation-based leakage).
- The Tran et al. work is a Lagrangian-dual fairness method, not simply "adaptive clipping".

---

## 7. Final Verdict & Checklist of Actionable Revisions

**Verdict: Major revision.** The idea is reasonable, but the claims outrun the theory, the experiments, and the clinical evidence.

### Must-fix (Critical)
1. Correct the Theorem 1 proof: use the exact subsampled-Gaussian RDP (or the accountant), not q²λ/(2σ²). Report T, N used in the accountant, q, sampler (Poisson?), optimiser, and architecture.
2. Resolve the prior-leakage contradiction: use truly public priors, or account for private estimation of p_a and |S_a|.
3. Specify the full privacy budget: preprocessing, SWD data source (C12), hyperparameter tuning, and ε_conf. Report ε_total. Replace single-σ runs with fixed target ε values (for example 1, 3, 8) and show an ε sweep.
4. Repair the conformal proposition: include the Laplace tail in the buffer, prove the mechanism's DP, and handle the data-dependent noise scale. Compare against exact split-conformal group-conditional quantiles, which are far less conservative. Report per-group calibration sizes, marginal coverage, and set sizes. Show that sets are not trivially {0,1}.
5. Explain or fix the ablation impossibilities: DP-GCCP cannot change AUC/ECE/ΔSP. Re-run with shared trained models. Check the w/o SC-DP ΔEO = 0.177 coincidence. Explain why every ablation beats the full model in AUC.
6. Add the missing baselines: DPSGD-F, Tran et al., Esipova et al., and DP + simple tabular models, plus a matched "DP-EDL + SWD" arm and a non-private TRACER-Med.
7. Replace n = 3 with repeated CV (for example 5×5 or 10 seeds), report CIs and paired tests, and report threshold-free fairness curves. Acknowledge that test groups have single-digit positives.
8. Replace or heavily qualify the synthetic EHR cohort: describe the generator, validate it, drop the compliance claims drawn from it, or use a real credentialed dataset (MIMIC-IV, eICU) with real group differences.
9. Report all promised metrics (PR-AUC, per-group ECE, subgroup AUC gap, marginal coverage, set size, faithfulness rank correlation), and report Pima ΔEO and coverage honestly in text and abstract.

### Should-fix (Major)
10. Use a correct aleatoric/epistemic decomposition (expected entropy and mutual information). Validate vacuity via OOD or error-detection experiments, and tone down "calibrated uncertainty".
11. Make SWD class-conditional, or justify unconditional alignment against the paper's own EO argument.
12. Justify Channel A (ln α_ŷ versus the margin), analyse Channel A/B correlation, run sanity checks, and use a public or DP baseline x⁰. Add a DCDA ablation, an IG-on-softmax or SHAP comparison, and a clinician-facing evaluation.
13. Fix the fairness theory: attribute SP/EO incompatibility correctly, drop "EO avoids miscalibration", and justify (or remove) SP as a clinical target for sex and age.
14. Rework the clinical framing: `ca`/`thal` leakage, the Pima age split (verify 46%/54%), imputation inside CV folds, SOFA redundancy, and endpoint definition.
15. Replace the abstention rule with calibrated, evaluated policies (risk–coverage, abstention rate, decision-curve analysis).
16. Rewrite the regulatory section: "supports documentation of", not "satisfies". Correct GMLP principle mapping, PCCP title, and EU AI Act/Section 1557 specifics against current (2026) status.
17. Upgrade the MIA audit (LiRA, canaries). Reword the "corroborates" language, and fix the stated MIA range (DP-SGD 0.571 on Heart).

### Polish (Minor)
18. Unify notation (z, ŷ, B′, coverage-gap naming, Rényi spelling). Define ℓ_i, y_ik, λ_reg, annealing, M, IG steps.
19. Fix tables: split or reformat, bold math headers, consistent precision, D = 12 versus 13, and Table 3 caption "across benchmarks".
20. Reorder figures, add ΔEO and per-ε panels, and normalise/explain the radar axes.
21. Update references to 2025–2026, verify the unverifiable ones, and fix misattributions.
22. Retitle sections ("Discussion" vs "Results"), cut "Trilemma" or define it formally, rename "Adaptive", and align the Elsevier declarations (AI use, ethics, ORCID).

I can cross-check specific table values, group sizes, ε, and conformal set sizes against `experiments/results/` and `src/`. I can also draft a point-by-point response outline if you want to work from this review.
