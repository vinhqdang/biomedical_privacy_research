# Referee Report: TRACER-Med (submitted to *Artificial Intelligence in Medicine*)

I audited the manuscript as supplied. I recomputed the key quantities by hand: the clipping weights, the τ_a buffer, the Dirichlet identities, the percentage reductions, and the Table 1 bolding. I did not see `references.bib`, so reference comments are limited to what the text shows.

## 1. Overall Assessment & Summary of Contributions

**Summary.** The paper combines five components in one pipeline:
- an evidential Dirichlet network;
- group-specific DP-SGD clipping (REF-DP) with weights derived from the Dirichlet Fisher metric;
- a per-sample class-prototype loss (PC-PAT);
- a Laplace-histogram group-conditional conformal calibrator (DP-GCCP);
- dual-channel integrated-gradients attribution (DCDA).

It is evaluated on Statlog Heart, Pima, and a synthetic EHR cohort.

**Assessment.** The motivation is clinically relevant, and the writing is organised. Several design choices are sound:
- Separating the evidential, fairness, conformal and XAI layers is clean.
- The per-sample (non-batch-coupled) fairness loss is the right instinct for DP.
- Flagging "vacuity collapse" of the coverage gap is a good reporting practice.
- The cautious ε discussion for N=270 and the acknowledgement of invasive Heart features are welcome.

In its current form, however, the central claims are not supported:
1. The conformal PAC guarantee is vacuous at every cohort size used in the experiments. The reported sets come from an unprincipled, un-accounted shrinkage heuristic.
2. The headline fairness win (Δ_SP on Heart) conflicts with the paper's own argument that SP is the wrong metric. The ablation shows each component worsens Equalized Odds, ECE, AUC and coverage gap.
3. REF-DP does not do what the text says (see §3.3).
4. There are verifiable mathematical errors: the mutual-information sign, a "diverging" trace of a bounded quantity, and a misleading RDP expression.
5. The experiments are underpowered, the baselines are weak, and some table entries look duplicated.

Recommendation: **major revision**. It is not publishable without the changes in §7.

## 2. Major Strengths

- It integrates privacy, fairness, uncertainty and explanation, which is a topical and clinically motivated combination.
- The privacy engineering of the training step is mostly correct. The loss is per-sample, LayerNorm is used rather than BatchNorm, and group-dependent clipping is still bounded by C_max. This makes the sensitivity argument valid.
- The Sherman–Morrison inverse of the Dirichlet Fisher matrix is correct. The Dirichlet KL, the Bayes loss, the logit gradient, and the expected-entropy formulas are also correct.
- The Laplace/Bernstein algebra for the cumulative-noise term is correct as algebra: t ≤ 2b√(M·L) + 2bL with L = ln(4M/β).
- The unusually explicit discussion of the SP versus EO incompatibility, base rates, and the Heart prevalence gap (about 52% versus 29%, which I verified) is good.
- The paper reports the vacuous-set artefact and excludes it from bolding.
- Limitations are listed candidly, including the inference-time attribute requirement and the ε=15.36 caveat.
- The regulatory context is mostly current: FDA PCCP (Dec 2024), GMLP, and EU AI Act Articles 9–15.

## 3. Critical Methodological & Mathematical Inquiries

### 3.1 Evidential Dirichlet formulas (Section 4.1)

- **Sign error in the mutual information.** The expression
  I = Σ p̂_k(ψ(S+1) − ψ(α_k+1) − ln p̂_k)
  is wrong.
  - The correct form is I = Σ p̂_k(ψ(α_k+1) − ψ(S+1) − ln p̂_k).
  - The paper's own aleatoric line gives H − E[H] = H(p̂) + Σ p̂_k(ψ(α_k+1) − ψ(S+1)).
  - For large evidence, ψ(α_k+1) − ψ(S+1) ≈ ln p̂_k, so the correct I → 0. The printed version tends to −2 Σ p̂_k ln p̂_k, which is large.
  - Please check that the code matches the correct form. Channel B (DCDA) depends on it.
- **The "diverging" curvature claim is false.** The text says Tr(𝒢) "diverges" as α_j → 1 and, in the same paragraph, gives ψ₁(1) = π²/6.
  - Since α_k ≥ 1 always, ψ₁ ≤ 1.645 and Tr(𝒢) is bounded.
  - The claim that EDL gradients "depend on inverse evidence concentrations" is overstated for the same reason: the logit gradient is bounded by ψ₁(1)·σ(z) ≤ 1.645.
  - Tr(𝒢) is also not a "scalar curvature"; that term refers to the Riemannian curvature scalar.
- **Prior art on EDL is under-weighted.** Bengs et al. and follow-up work (e.g. Jürgens et al., ICML 2024) show that EDL's second-order and epistemic estimates do not faithfully represent epistemic uncertainty. The paper then builds Channel B and the abstention rule on this quantity. Add a validation (OOD detection, or the known ground truth in the simulator).
- **Notation collisions:**
  - 𝐞 is the evidence vector, 𝐞₁/𝐞₂ are basis vectors, and *e* is the epoch.
  - δ is the privacy parameter, the Kronecker delta, and the Dirac measure.
  - 𝒮_a (calibration subsets) collides with S (evidence strength).
  - Δ is used both for sensitivity (Δ₁) and for disparity (Δ_SP, Δ_EO).

### 3.2 RDP theorem (Theorem 1)

- **The closed form is not a valid guarantee.**
  - The displayed ε(λ) = Tq²λ/(2σ²) + O(·) uses the unsubsampled-Gaussian constant (λ/2σ²) multiplied by q².
  - The commonly cited small-q leading term for the subsampled Gaussian is about 2q²λ/σ² (Abadi, Mironov 2019).
  - Those expansions require σ ≳ 1 and moderate orders.
  - Here σ = 0.8 and q ≈ 0.18 on Heart (32 over about 180 training records), so the O(q³λ²/σ³) remainder is not negligible.
  - Replace the formula with a statement that the guarantee is the numerical RDP accountant output, and drop the display.
- **The conversion is not "exact".** ε = min_λ {ε(λ) + ln(1/δ)/(λ−1)} is the classical, loose conversion. Tighter conversions exist (Balle et al. 2020, Canonne et al. 2020). It is an upper bound.
- **Wording:** "preserves (ε,δ)-DP via optimal order selection" is fine. Elsewhere, "(ε_tr,δ) holds with probability 1−δ" is a misreading of (ε,δ)-DP. Remove it.
- **Components outside the accounting.** The formal claim covers only the SGD loop. None of the following is accounted for:
  - (a) hyperparameter selection (σ, C₀, λ_fair, epochs, lr, the 15 and 0.96 constants), despite citing Papernot & Steinke;
  - (b) the priors p_a and S̄_a. 0.678/0.322 and 0.544/0.456 are the datasets' own group proportions, not "external epidemiological registries";
  - (c) median imputation and feature standardisation on training data;
  - (d) q̂_marg (see §3.4);
  - (e) n_a, used in τ_a and λ_a, which is not public under add/remove adjacency.
  
  End-to-end DP is therefore not established, and the abstract's language should reflect that.
- **Adjacency mixing.** Section 3.2 declares add/remove adjacency. The conformal module then uses "stratified bounded replacement" (Δ₁ = 2), while the Introduction, contributions and abstract state Δ₁ = 1 ("exact"). The Δ₁ values and adjacency notions must be unified. The parallel-composition argument for max(ε_tr, ε_conf) additionally needs a data-independent partition. State this and treat θ as public post-processing input to the calibrator.
- **Sampling.** The experiments say "batch size 32", while the theory assumes Poisson sampling. Name the DP library (Opacus?) and confirm Poisson sampling is actually used.

### 3.3 REF-DP: what the clipping weights actually do

I recomputed the weights from the stated priors:

| Dataset | Majority clip C₀* | Minority clip C₁* = C_max |
|---|---|---|
| Heart | ≈0.80 | ≈1.33 |
| EHR | ≈0.77 | ≈1.39 |
| Pima | ≈0.97 | ≈1.03 |

- **There is no SNR amplification for the minority.** Noise std is σC_max ≈ 1.07 (Heart). The minority's SNR is C₁*/(σC_max) = 1/σ, which is identical to the uniform-clipping baseline.
  - The majority's SNR drops by about 40%, and absolute noise rises 33–39% over C=1.
  - So REF-DP levels down the majority and does not amplify minority SNR, contradicting "selectively amplifying the effective SNR of underrepresented patients".
- **On Pima, REF-DP is effectively uniform clipping.** The groups are 54/46 and the weights are within about 6% of each other. The "minority" there is 45.6% of the cohort. Any Pima differences attributed to REF-DP are noise.
- **The Fisher geometry never enters the algorithm.**
  - The Riemannian norm and natural gradient are defined but never used. Clipping is Euclidean in parameter space.
  - The Fisher metric supplies only a scalar via the approximation S̄_pop/S̄_a, using hand-picked S̄ values. "Prior expected evidence 2.4 for women due to referral bias" has no empirical basis.
  - For K=2, S̄ ≈ 2.4–3.5 implies total evidence of about 0.4–1.5.
  - "Derive" and "restores information-geometric invariance" are therefore unsupported. No theorem relates C_a* to disparity or to invariance.
- **The premise is not tested.** The paper never shows per-group gradient norms or per-group clip fractions to demonstrate that minority gradients are over-clipped.
- **The weighting factor** (1/(2p_a))^{1/2} hard-codes |𝒜|=2 and is unjustified beyond heuristics.

### 3.4 DP-GCCP: the theorem is vacuous in the experiments, and the proof has gaps

- **Vacuity threshold.** Plugging in the paper's constants (Δ₁=2, M=20, ε_conf=1, β=0.05):

  | n_a | τ_a |
  |---|---|
  | 50 | ≈1.67 |
  | 1,000 | ≈0.12 |
  | about 1,300 | ≈0.097 |

  - 1−α+τ_a ≤ 1 requires τ_a ≤ 0.10, so n_a ≳ 1,300 per cohort.
  - Section 5 says the collapse occurs for n_a < 50. That is wrong by more than an order of magnitude.
  - Heart and Pima are vacuous. The EHR minority (about 30% of a calibration split of at most a few thousand) is vacuous or borderline.
  - Every reported set therefore comes from the shrinkage heuristic q̂_a^reg, which has **no guarantee**.
  - The abstract and highlights ("Establishes PAC conformal bounds", "PAC concentration buffers") do not describe what was evaluated.
- **The shrinkage constants are ad hoc.** λ_a = n_a/(n_a+15) and the 0.96 cap are unjustified. They are probably tuned on the evaluation data.
  - With q̂ capped at 0.96, class k is included whenever p̂_k ≥ 0.04. Given EDL probabilities (low evidence, p̂ roughly 0.2–0.8), essentially every set is {0,1}. This matches the reported sizes of 1.80–1.87, i.e. 13–20% singletons.
  - Calling this "non-vacuous" is a stretch. ERM yields about 1.5.
  - The cap prevents vacuity by construction and not through merit.
- **q̂_marg is undefined.** If it is computed non-privately, it breaks the DP claim. If it is privately computed, extra budget must be accounted for.
- **Proof gaps:**
  - (i) The mechanism uses c̃ = max(0, c+ξ). The truncation makes the noise non-zero-mean (about +b/2 per empty bin; 20 bins × 1 ≈ 20 spurious counts, large for n_a ≈ 30–100). The proof treats the cumulative noise as Σξ_j, which is no longer true. |c̃−c| ≤ |ξ| only gives Σ|ξ_j| ≈ Mb, not the √M rate.
  - (ii) F̃ is normalised by max(1, Σc̃), not n_a. The error F̃−F̂ ≠ Z_m/n_a, because the ratio term is ignored.
  - (iii) "Probability over the random draw of calibration sample" should include the Laplace noise.
  - (iv) Bernstein's constants for Laplace sums should be cited or derived.
- **Comparison fairness.** The table does not say whether baselines use the same (private, group-conditional) calibrator. ERM's 1.49 set size suggests non-private marginal split-CP. If so, set size and coverage gap are not comparable across rows.
- **Coverage gap is noise-dominated.** On Heart, the test fold has about 90 patients and about 29 women. Sampling SD of per-group coverage is ≈ √(0.09/29) ≈ 0.056, and differences of 0.02 are well under that floor. Report per-group coverage with CIs and marginal coverage (neither is in the table).

### 3.5 PC-PAT and the SWD claims

- **Not a fairness mechanism.** ℓ_PAT(x,y) = ½‖h−μ_y‖² never uses the demographic attribute *a*. It is a center loss with learned centers (Wen et al. 2016). It pulls both groups to the same class centre, whether or not they differ.
- **The Wasserstein bound.**
  - It is correct but trivial: W₂(P,δ_μ)² = E‖h−μ‖² = 2E[ℓ_PAT]. The "Monge–Kantorovich duality" citation adds nothing.
  - "Contracts the transport distance to zero" requires representation collapse, which destroys the information the evidential head and Channel A/B read.
  - The bound is loose and gives no guarantee on Δ_EO.
  - Learnable μ₀, μ₁ can collapse to each other (no separation regulariser), and unit-norm prototypes sit on a different scale from LayerNorm'd features.
- **The "strict per-sample independence" is trivial.** Any per-sample loss has it. It is not a theorem-level contribution, and "resolving sensitivity leakage" applies only to a batch-coupled SWD baseline that was never run privately.
- **The empirical evidence points the other way.** In the ablation, removing PC-PAT improves Δ_EO (0.123 vs 0.158), ECE, AUC and the coverage gap.

### 3.6 DCDA (Proposition 3)

- Completeness is the Fundamental Theorem of Calculus. "Implementation invariance" and "Linearity and Symmetry" are asserted without proof. Implementation invariance is trivial for IG, and the "additive combination of evidential components" sentence is vague.
- LeakyReLU makes the integrand piecewise linear. Gauss–Legendre with 50 nodes is not guaranteed to converge fast, and the "<10⁻⁴" bound is empirical and should be stated as such.
- The infidelity values (≈0.003 for DP models vs 0.012 for ERM) probably reflect flatter functions or smaller gradient scale, since infidelity is scale-dependent. They do not show "stable attributions despite DP training".
- Channel B uses I(x⁰) ≠ 0, so attributions can be negative. Say what negative "doubt" means.
- Validate attributions against the **known** generative ground truth in the synthetic EHR.

## 4. Clinical & Biomedical Accuracy

**Reference baselines:**
- **Statlog Heart.** Anchoring `cp` at "category 0 (typical asymptomatic baseline)" is clinically wrong. In Cleveland/Statlog the asymptomatic chest-pain type is the *highest*-risk category.
  - The coding is 1–4 in OpenML 53, but 0–3 in the quoted text; the two are conflated, and "typical angina" is mislabelled as "asymptomatic".
  - Nominal variables interpolated along a line (cp = 1.5) are non-physical, which is exactly the objection the paper raises against medians.
  - The baseline list omits age, restecg, slope and thal.
- **Pima.** "Normal fasting baseline 100 mg/dL" for 2-hour OGTT glucose conflates two tests. The normal 2-h threshold is <140 mg/dL, and 100 is the fasting prediabetes cutoff. The baselines for skinfold and pedigree function are missing.
- **EHR.** The reference values are reasonable (P/F 450 ≈ 95/0.21 ✓, MAP 80 for SBP 120 ✓). However:
  - creatinine, platelets and WBC have sex- and age-specific reference ranges, and a single "healthy" point conflicts with the fairness narrative;
  - a "healthy" reference is not a patient's counterfactual.

**EHR cohort realism:**
- A 42% outcome prevalence for "ICU transfer or 30-day mortality within a 24-hour prediction window" is unrealistic. Ward deterioration prevalence is typically a few percent. The 24-h horizon and 30-day mortality are also incompatible.
- Lactate, ABG-derived P/F, bilirubin and platelets are ICU variables, usually not routinely available on wards. Missingness is informative (measured only in sicker patients), and none of this is modelled.
- SOFA is a deterministic function of MAP, platelets, bilirubin, creatinine and P/F (plus GCS and vasopressors), so including it alongside its components is collinear and leaks.
- The protected attribute is unspecified ("demographic cohort"). The central clinical mechanism of EHR bias, differential measurement quality (e.g. pulse-oximetry bias by skin tone), is absent.
- Baseline disparities are almost nil: ERM has Δ_SP = 0.011 and Δ_EO = 0.041. There is nothing for the method to fix, which makes the EHR result uninformative about fairness.
- Real MIMIC/eICU-derived parameters require the PhysioNet credentialed-use terms. State how the distributions were derived and shared.

**Clinical workflow (Section 7.2):**
- The rule u < 0.20 means S > 10 for K=2. The REF-DP priors put S̄ at about 2.4–3.5, so u ≈ 0.6–0.8 for almost everyone.
  - Combined with near-full prediction sets, the system would abstain on nearly all patients.
  - The abstention policy is also never evaluated (no risk–coverage curve, no PPV at action).
- Heart (invasive-test features) is a diagnostic-confirmation setting, so the "co-pilot" story does not transfer.
- Per-group FNR/TPR (the clinically critical quantity named in the Introduction) is never reported with CIs. With about 8 female positives per Heart test fold, one patient moves TPR_1 by 0.125.
- Using the protected attribute at inference to choose thresholds is disparate treatment. EU AI Act Art. 10(5) permits processing of special-category data for *bias detection and correction* under safeguards. It does not "explicitly authorize" inference-time use. 45 CFR §92.210 requires identifying and mitigating discrimination risk from decision-support tools, and does not authorize such use. Rephrase.

**Regulatory text:**
- The claim that TRIPOD+AI and FUTURE-AI recommend condition-specific error parity "rather than demographic parity" is stronger than those documents say. They require subgroup reporting.
- FUTURE-AI was published in the BMJ in 2025 (cite the journal version).
- Re-check the EU AI Act application timeline for Annex I high-risk obligations, since amendments were under discussion.

## 5. Statistical & Experimental Soundness

- **Underpowered design.** Three folds, N=270 (about 90 test patients, about 29 women, about 8 female positives) is far too small. The reported SDs (e.g. Δ_EO ±0.01 for DP-EDL on Heart, MIA ±0.00) are implausibly small unless folds or seeds are correlated. State precisely what the "3 seeds" are (folds? seeds? both?), and use repeated stratified CV (e.g. 10×5), bootstrap CIs, and paired tests.
- **The headline win is not a win under the paper's own criterion.**
  - Heart: DP-EDL beats TRACER on AUC, ECE, Brier, Δ_EO (0.115 vs 0.158) and MIA. TRACER wins only Δ_SP (and the non-vacuous cov-gap row, by default).
  - Pima: differences in ECE (0.117 vs 0.119), Brier, SP and MIA are far below one SD. DP-EDL has the better AUC, and its non-vacuous coverage gap (0.009) beats TRACER's (0.025).
  - EHR: TRACER wins ECE and MIA by 0.001. Δ_SP is worse than ERM and DP-EDL, and the coverage gap is among the worst.
  - ECE under DP on EHR is 0.145 versus 0.043 for ERM, so calibration under DP is not fixed on the large dataset.
  - Δ_SP is promoted as the headline even though §3.3 argues it is clinically inappropriate. A lower Δ_SP than the base-rate gap is not a virtue either.
- **Overstated textual claims:**
  - "All private models … ROC-AUC 0.746–0.748 and Brier 0.225–0.226" is false, since Standard DP-SGD has AUC 0.743 and Brier 0.254.
  - "TRACER … minimal MIA vulnerability 0.505" ignores DP-EDL's 0.506.
  - On Heart, TRACER's MIA (0.565) is the worst among private models.
  - "Harmonization of the Trilemma" is unsupported, since no model dominates.
  - "Mitigates demographic disparities" in the abstract is not supported for EHR.
- **The ablation is internally inconsistent and mostly unsupportive.**
  - Table 2 shows w/o REF-DP, w/o PC-PAT and w/o DP-GCCP each improving ECE, AUC and Δ_EO (0.132/0.123/0.118 vs 0.158), with only Δ_SP better for the full model.
  - w/o REF-DP gives Δ_SP = 0.318 ± 0.03, identical to Standard DP-SGD. That model still has EDL and PAT, so a copy-paste error is plausible.
  - w/o PC-PAT and w/o DP-GCCP share the same cov-disparity 0.008 ± 0.011, which looks like duplication.
  - Swapping a post-hoc conformal layer cannot change AUC, ECE, SP or EO if the model is fixed. The 0.881 vs 0.875 (and related) differences imply different training runs. They are then noise, which also exposes SDs of 0.02–0.04.
  - The text claims set-size results (1.87→1.96) with no table column, and says "across the benchmarks" while only Heart is shown.
  - "w/o REF-DP widens the gap to 0.037 ± 0.052" is well within noise. Item 3 shows the marginal CP's gap (0.008) is smaller than DP-GCCP's, which contradicts the narrative.
  - The "essential for equalizing gradient representations" claim is not measured.
- **Baselines.**
  - Missing: Fair/DP baselines named in Related Work (DPSGD-F, Tran et al., Lowy et al.'s stochastic DP-fair learning, Esipova), post-processing EO (Hardt), Mondrian/RAPS conformal, and strong tabular baselines (logistic regression, gradient boosting, DP-logistic regression).
  - Fair-ERM is non-private, so the paper has no private fairness comparison. Tuning protocols are unspecified.
  - Private models beat non-private ERM on Heart by 3 AUC points. This suggests untuned or over-fit non-private baselines and inflates the apparent "benefit" of DP regularisation.
- **MIA audit.** MIA AUCs of 0.50–0.56 on tiny test sets with shadow models are within sampling noise. At ε=15.36 the audit says little. Fig. 4 uses 1−MIA, which compresses an axis to [0.43,0.50] and makes area comparisons meaningless. Radar areas also reward vacuous coverage gaps (DP-EDL scores a perfect "1−Cov. Gap").
- **Reproducibility.** The "≈4,200 parameters" figure does not match 64/32 hidden dimensions: I count about 3.2k for Heart. It is also unclear how calibration data are split inside each 3-fold CV (shrinking the training set), whether hyperparameters were tuned on test folds, and which DP library and version were used.

## 6. Minor Issues, Writing, LaTeX & Micro-Edits

**Front matter**
- The submitted title ("A Unified Trustworthy … Dual-Channel Explainability") differs from the source title. Align them.
- The abstract is overlong, number-dense, and omits the unfavourable Δ_EO and vacuity caveats.
- Check keyword count (8) against the journal's limit (usually ≤6).
- Highlight 3 ("Proves strict per-sample record privacy") and highlight 4 overclaim.
- AI declaration: it names "Claude 3.7 Sonnet". Verify the tool names and versions are accurate. This also conflicts with your project rule of not mentioning a Claude identity. Elsevier requires disclosure, so decide deliberately.

**LaTeX**
- Backtick-quoted identifiers like `` `fbs` ``, `` `cp` ``, `` `ca` ``, `` `thal` `` render as ‘fbs‘. Use `\texttt{fbs}`.
- `\bar{|\mathcal{C}|}` renders badly. Use `\overline{|\mathcal{C}|}`.
- `\mathbf{1.80 \pm 0.15}` does not bold `\pm`. Use `\bm{...}`.
- Missing blank line before `\section{The TRACER-Med Architecture}` (it follows `…clinical condition.` directly).
- `\documentclass[3p,12pt]` is an unusual combination. The long displays (τ_a in the theorem and in the algorithm line) and the 10-column `\resizebox` table are likely to overflow or shrink below legible size.
- The table footnote is built with `\smallskip\newline\raggedright`. Use `threeparttable` or a `minipage`.
- Table cells mix 2-decimal and 3-decimal SDs. Bold marks values within one SD of ties. Standardise.
- In Proposition 2, `x ~ P_{0,y}` is used for a latent-space distribution of h. Write `h ~ P`.
- Algorithm: `q̂_marg`, Δ₁, M_bins and the constants 15/0.96 are missing from REQUIRE. `\bar{S}_a` and `p_a` are listed as "epidemiological priors" but are data-derived.
- Define `\Delta_1` once and consistently (see 3.2).

**Wording and consistency**
- "Trilemma" is never formalised. "Tension" is more accurate. Several "trade-offs" are only cited.
- "Chouldechova/Kleinberg proved predictive parity (within-group calibration)…" conflates predictive parity (PPV) with calibration.
- Farrand et al. is cited for calibration inflation under DP. It addresses accuracy disparity. Bu et al.'s work on DP and calibration is the apt citation.
- `patel2022towards` is cited for two unrelated claims (explainability and privacy auditing). Verify.
- Check the Esipova et al. year (ICLR 2023). Few 2025–2026 works appear in the cited text, so update the literature as promised.
- "Standard DP-SGD" and "EDL" are described as "overparameterized"; with about 3k parameters on about 180 samples, specify.
- Pima uses *age* as both a feature (D=8) and the protected attribute. Heart drops *sex* from the features (D=12). Treat these consistently.
- The Fig. 2 caption says EHR only, but the text says "across the evaluated benchmarks".
- Section 6.2 reports ε as ε_tr only. Always state the pair (ε_tr, ε_conf) in the table caption.
- In the ethics statement, "exempt from IRB oversight" is for the institution to determine. Say "IRB review was not required". Add a note on data-governance concerns for the Pima dataset.
- The 18.2% and 31.4% figures use different denominators (reduction vs DP-EDL; increase over TRACER). State which.
- "Closed-form" and "exact" are overused (RDP conversion, ℓ₁ sensitivity).

## 7. Final Verdict & Checklist of Actionable Revisions

**Verdict: Major revision.** The idea of a trustworthy private clinical pipeline is worthwhile. The current evidence does not support the "PAC guaranteed", "mitigates disparities", or "harmonizes the trilemma" claims.

**Must fix (blocking)**
1. Correct the mutual-information sign (§3.1). Verify the code and re-run Channel B.
2. Remove the "Tr 𝒢 diverges" and "inverse-evidence gradient" claims. Define the curvature terminology correctly.
3. Replace the closed-form RDP expression with the accountant output and a correct citation. Drop "exact".
4. Fix the DP-GCCP theory:
   - redo the proof for truncated Laplace counts and the normalised CDF;
   - unify the adjacency and Δ₁;
   - state the n_a ≳ 1,300 non-vacuity requirement.
   Then either (a) put the shrinkage heuristic into the theory, define and account for q̂_marg privately, and justify 15/0.96, or (b) present results with and without it and remove PAC claims about the reported sets. Reduce M_bins, use pooled or hierarchical calibration, or use a larger calibration share to make the bound non-vacuous.
5. Retract or rederive the REF-DP SNR claim (§3.3), and show per-group gradient norms and clip fractions. Justify S̄_a from data-independent sources, or learn them privately and account for them.
6. Make an honest fairness headline. Lead with Δ_EO and per-group TPR/FPR (with CIs), reconcile with the SP discussion, and stop treating lower Δ_SP as a win.
7. Rebuild the ablation: all benchmarks, matched seeds, same model for the conformal-only ablation, set-size column, and no duplicated numbers. Check the 0.318 ± 0.03 and 0.008 ± 0.011 entries.
8. Add baselines: DPSGD-F, Tran et al., post-hoc EO, Mondrian/RAPS conformal (private and non-private) under an identical protocol, logistic regression/GBDT, and DP-logistic regression. Report tuning budgets.
9. Replace 3-fold results with repeated CV, CIs and paired tests. Report marginal and per-group coverage and per-group TPR/FPR.
10. Account for or disclose everything outside the DP loop (imputation, scaling, priors, hyperparameter tuning, n_a, q̂_marg), and soften "end-to-end".

**Should fix**
11. Redesign or justify the EHR simulator: realistic prevalence and horizon, ward-appropriate features, informative missingness, group-dependent measurement bias, and an explicit protected attribute. Provide a ground-truth check for DCDA.
12. Correct clinical baselines (cp coding and baseline; OGTT labelling; complete baseline lists; sex/age-specific ranges).
13. Reconcile u < 0.20 with the actual evidence scale. Evaluate selective abstention (risk–coverage).
14. Rework the PC-PAT narrative (center loss, no use of *a*), add a collapse regulariser, and add an empirical Wasserstein/EO diagnostic.
15. Correct the regulatory language (Art. 10(5), §1557, TRIPOD+AI/FUTURE-AI) and the timeline.
16. Update references (FUTURE-AI BMJ 2025; DP calibration; DP-fair learning in medical imaging; recent private conformal prediction; EDL critiques; 2025–2026 work), and verify `farrand2020neither`, `patel2022towards`, `esipova2022gradient`.

**Polish**
17. Fix all LaTeX items in §6. Compile and check overfull boxes.
18. Resolve notation collisions (𝐞, δ, 𝒮/S, Δ).
19. Rename "trilemma", shorten the abstract, and align the title.

I can turn this into a point-by-point revision plan or implement the manuscript fixes (LaTeX, tables, claims) if you want.
