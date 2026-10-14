# Referee Report, Round 2

**Manuscript:** TRACER-Med (*Artificial Intelligence in Medicine*, elsarticle)
**Recommendation: Major Revision**

I reviewed the `.tex` source only. I did not see the figures or the raw experiment outputs, so my checks of Round 1 item 7 cover the text, tables and captions.

## 1. Summary

The presentation is much improved. The dataset is now labelled as simulated, the notation is cleaner, the tone is more measured, and the Elsevier front and back matter is in place. Several central problems remain, though:

- The differential-privacy (DP) conformal step is mathematically incorrect.
- The Theorem 1 proof has gaps.
- The full model's privacy claim does not cover the Sliced Wasserstein Distance (SWD) component.
- Many headline numbers in the abstract and conclusion do not match the paper's own tables.
- The experiments often contradict the narrative the paper builds on them.

## 2. Audit of Round 1 requests

| # | Request | Status |
|---|---|---|
| 1 | Correct Theorem 1 with C_max and fixed priors | **Partially addressed.** C_max and fixed priors are there, but the proof has three defects (§3.1). |
| 2 | DP-GCCP with buffer and explicit composition | **Formalised, but wrong.** See §3.2. |
| 3 | Label the dataset as simulated | **Addressed**, with caveats. "Standardized clinical inpatient benchmark" still appears in §5.1. There is no citation for MIMIC-IV or eICU and no generator specification. |
| 4 | Disambiguate notation | **Mostly addressed.** Residual overloads are listed in §5. |
| 5 | Remove promotional language | **Partially addressed.** See §5. |
| 6 | Elsevier requirements | **Present, with defects.** Some highlights appear too long and the abstract may exceed the word limit (§5). |
| 7 | Tables and figures match results | **Not achieved.** See §3.3. |

## 3. Major concerns

### 3.1 Theorem 1 is still not sound

1. **Adjacency mismatch.** The proof says Δ₂ ≤ C_max, but it also says a patient is *replaced*.
   - Under replace-one adjacency, the sum changes by ḡ_j′ − ḡ_j, so Δ₂ ≤ 2C_max. That would halve the effective noise multiplier to σ/2.
   - The bound C_max holds only under add/remove adjacency.
   - State which adjacency is used and make the proof, the accountant and the code consistent.
2. **The RDP bound is not valid "for any order λ > 1".** q²λ/(2σ²) per step is a small-q, bounded-λ approximation (compare Abadi et al. and Mironov et al. 2019).
   - Either restrict λ and σ, or use the exact subsampled-Gaussian RDP computation.
   - Also confirm the implementation really uses Poisson sampling. Algorithm 1 shows mini-batches of fixed size normalised by |B|.
3. **The priors p_a must be public.** If they are estimated from the training data (for example the 32% and 29% minority shares), they are data-dependent and leak. Say how they are obtained.

The "equalizes gradient SNR" claim in the Highlights and Contribution 3 is not proved. Per-sample SNR is C_a/(σC_max), which equals 1/σ only for the group with C_a = C_max. The mechanism rebalances clipping; it does not equalise SNR. "Unconditional" in Contribution 3 should also go.

**Privacy of the full model.** SWD couples samples within a batch and is applied "during non-private training". Per-sample gradients are then not well-defined, so Theorem 1 does not cover the full TRACER-Med that is evaluated. Either SWD is outside the DP guarantee (so the model is not DP), or it needs a DP treatment. SWD is also absent from Algorithm 1, and λ_fair and the "demographic parity loss" used by Fair-ERM are undefined.

**Missing privacy parameters.** No ε, δ, q, T, number of epochs, C, γ or learning rate is reported anywhere. With N = 270 and σ = 0.8, ε may be vacuous. Please add ε–utility curves.

### 3.2 DP-GCCP is incorrect and has no coverage theorem

1. **Wrong sensitivity.** Lap(1/(|S_a|ε)) added to the quantile *value* is not ε-DP. An order statistic of scores in [0,1] has worst-case sensitivity up to 1, not 1/n.
   - The 1/n sensitivity applies to the empirical CDF at a fixed threshold.
   - Use the exponential mechanism, a noisy-CDF/histogram approach, or add noise on the *level* scale, as in Angelopoulos et al.
2. **Scale mismatch.** The buffer τ_a = 2ln(10)/(nε) is added to a probability level, while the noise it is meant to dominate lives on the score scale.
3. **Buffer saturates.** With about 15–25 minority calibration scores and ε_conf = 1, τ_a ≈ 0.2–0.3. The quantile level then exceeds 1, the rule is undefined, and the clamp yields trivial sets.
   - The reported 97.1% marginal coverage at a 90% target is consistent with this.
   - Average set cardinality is promised in §5.3 but never reported.
   - Please also report the singleton, empty-set and full-set rates.
4. **Composition is loose, not "strict".**
   - Calibration and training use disjoint splits, so the per-record cost is max(ε_model, ε_conf), not the sum.
   - Parallel composition across groups also needs add/remove adjacency. Under replacement, a record can move between groups and cost 2ε.
5. **No coverage theorem.** With the noise and buffer, the finite-sample guarantee is only "with high probability" and is never stated. Per-group coverage with n_a ≈ 15–30 has a sampling standard deviation of roughly 5–7 points, so "gap ≤ 3.1%" is not achievable without much more conservatism.
6. **Group attribute at test time.** The set requires the protected attribute at inference. This is a possible disparate-treatment issue under the Section 1557 discussion in §6 and needs addressing.

### 3.3 Numbers in the text contradict the tables

- **Abstract AUC.** The abstract says ROC-AUC 0.85 ± 0.02. Table 1 gives 0.863, 0.826 and 0.674, and none of these is 0.85.
- **Abstract coverage gap.** The abstract says ≤ 3.1%. The table gives 4.5% (Heart), 5.8% (Pima) and 0.9% (EHR).
- **The "74%" claim.** It appears in the abstract, Contribution 3 and the Conclusion but cannot be derived from any table.
  - The best reductions I can compute are 27% in Δ_DP and 42% in Δ_EO against DP-EDL on Heart.
  - Pima shows no reduction.
  - Define "DP-induced disparity" precisely and show the computation.
- **The premise is not supported by the data.**
  - DP-SGD has *lower* Δ_DP than ERM on Heart (0.263 vs 0.283) and Pima (0.321 vs 0.409).
  - DP-SGD also improves ECE and AUC on Heart.
  - The trilemma motivation, and "analytical formulation" in Contribution 1 (there is no derivation in the paper), are therefore not demonstrated.
- **"Pareto-superior" is false.**
  - On Heart, TRACER-Med has the lowest AUC among the private models.
  - On Pima, its Δ_DP (0.361) and Δ_EO (0.258) are *worse* than DP-SGD and DP-EDL.
  - Its Pima coverage gap (0.058) is also far worse than DP-SGD (0.006) and DP-EDL (0.007).
  - The Results text reports only the favourable Pima numbers.
- **§6 governance claim.** The claim "Δ_DP < 0.10, Δ_EO < 0.10 across evaluated benchmarks" is false for Heart (0.232 Δ_DP) and Pima (0.361 and 0.258).
- **MIA range.** The text says private models fall in 0.500–0.547, but Standard DP-SGD on Heart is 0.571 ± 0.03. Non-private Standard EDL is 0.574 on Heart, so private and non-private are indistinguishable there.
- **MIA interpretation.** Low empirical MIA does not "corroborate" a formal guarantee. It is at best a weak audit.
- **Coverage evidence is inconsistent.**
  - The marginal-CP ablation gives a coverage gap of 0.046, against 0.045 for the full model. This shows no benefit from group conditioning.
  - Baselines using presumably marginal CP show smaller gaps than TRACER-Med.
  - The text attributes the 0.114 gap to "omitting group-conditional calibration (in the w/o SC-DP ablation)". That ablation changes clipping, not conformal calibration. The 0.114 ± 0.08 is also not distinguishable from 0.045 ± 0.05.
- **A post-hoc step cannot change the model.** Removing DP-GCCP is post-hoc and should leave ROC-AUC, ECE, Δ_DP and Δ_EO unchanged. Yet Table 2 shows AUC 0.863→0.890, ECE 0.077→0.097 and Δ_DP 0.232→0.274. Either the ablation retrains (so these differences are seed noise, undermining all ablation conclusions) or there is an error. The claim that DP-GCCP "provides essential post-hoc calibration equity" for Δ_DP is unsupported.
- **Ablation selectively quoted.** "w/o SC-DP" has *better* Δ_DP (0.220) and AUC (0.885) than the full model, and "w/o SWD" has coverage disparity 0.000. Only the favourable differences are discussed.
- **Table 2 bolding.** Δ_DP (0.220 < 0.232) and coverage disparity (0.000 < 0.045) are marked wrongly.
- **Possible copy error.** Table 2 shows Δ_EO = 0.177 for w/o SC-DP, identical to Standard DP-SGD in Table 1. Please check.
- **Table 1 bolding.** The caption says "best among private architectures", but bold is applied to non-private Standard EDL for ECE on Pima and EHR, and to ties.
- **Promised metrics never shown.** PR-AUC, subgroup AUC gap, average set cardinality and faithfulness rank correlation are listed in §5.3 and not reported.

### 3.4 Statistical and experimental design

- **Too few repetitions, no tests.**
  - "3 seeds" is ambiguous, because the setup says 3-fold CV with 3 seeds.
  - The table's standard deviations are over 3 values, so they are unreliable.
  - Many headline contrasts lie within one standard deviation (Δ_DP 0.232 ± 0.04 vs 0.263 ± 0.05, or vs ERM 0.283 ± 0.09).
  - With about 29 female test patients per fold, the standard error of a coverage estimate is roughly 0.056. A gap of 0.045 is within noise, and the absolute-value gap metric is biased upward.
  - Use at least 20 repeated splits with confidence intervals and paired tests.
- **Weak or missing baselines.**
  - ERM looks over-fitted (Heart ECE 0.199, and DP-SGD beats it on AUC).
  - Please add regularised logistic regression and gradient-boosted trees, which are standard on Statlog and Pima and often reach AUC of about 0.9 on Statlog.
  - Add the fair-DP methods the authors cite (Tran, Xu, Esipova), plain reweighting, and Mondrian CP.
  - State which conformal procedure each baseline uses. If baselines use non-private marginal CP, the comparison is not like-for-like.
- **The EHR benchmark carries no signal.** Δ_DP, Δ_EO and coverage gap are essentially identical (about 0.00–0.01) for every method, with AUC about 0.67–0.68. It cannot support any fairness or coverage claim. Yet Figure 2 and the DCDA case study rely on it.
- **Circular explanations.** The attribution case study recovers lactate and SBP thresholds from a simulator that the authors designed. This is not clinical validation. The simulator needs full specification, MIMIC-IV and eICU citations, and data-use statements.
- **DCDA is not shown to decouple.**
  - The two channels are functions of α_ŷ and S = Σα, so they are correlated by construction. No correlation analysis is given.
  - The baseline x⁰ is unspecified.
  - "Diagnostic evidence for disease presence" is wrong when ŷ = 0.
  - Infidelity values across EDL and softmax models are not comparable, since they explain different functions. Standard DP-SGD has lower infidelity than TRACER-Med.
- **Epistemic uncertainty is never validated.** There is no OOD, selective-prediction or abstention experiment. The u < 0.20 workflow threshold is arbitrary and untested, and the paper itself cites Bengs et al.
- **Hyperparameter selection.** Tuning on the same data without privacy accounting should be addressed.

## 4. Data and description checks

- The Pima age split looks inverted. The median age is 29, so ages ≤ 30 should exceed 50% of the sample, not 46%. Please verify.
- At p ≈ 0.46/0.54, the SC-DP weights are nearly 1, so SC-DP is almost a no-op on Pima. This may explain why it does not help there.
- EHR is listed as D = 15, but the described features sum to 13 (5 vitals + 6 labs + 2 indices).
- State whether the protected attribute is an input feature.
- Describe how the calibration split is carved from each fold.

## 5. Minor and editorial points

- **Residual hype.** "Cardinal pillars", "slashes", "successfully outperforming", "synergistic", "Pareto-superior", "guarantees balanced coverage" and "rigorous foundation". The "Accredited" in the acronym is meaningless.
- **Highlights.** My count puts three at 90, 91 and 86 characters against the 85-character limit. Two also assert unproved guarantees ("guarantees valid subgroup-conditional coverage", "equalizes gradient SNR"), and "Validated across…" overclaims for a simulated cohort.
- **Abstract.** It appears to exceed typical word limits for the journal; please verify.
- **Notation overloads.**
  - The subscript "DP" in Δ_DP clashes with differential privacy; consider Δ_SP.
  - C (clip) vs 𝒞 (conformal set); q (sampling rate) vs q̂ (quantile); S vs 𝒮; D vs 𝒟; z (logits) vs z (latent).
  - Dirichlet α is non-bold in the text but bold in the loss.
  - Δ_EO is a mean of gaps, not Hardt's max form. Say so.
- **Figure 1.** It is called a schematic overview but is a results radar. A real architecture diagram is missing.
- **§5.2.** "Eight baseline configurations" is unclear. There appear to be 5 baselines plus 3 ablations.
- **Empty prediction sets.** They are possible for K = 2 when q̂ < 0.5 and are not handled in the clinical workflow.
- **Regulatory section.** "FDA" is in the heading but no FDA content follows.
- **Related work.** It is dated: almost nothing from 2024–2026 on fair DP, DP conformal prediction or evidential models under DP. Please update.
- **Declarations.** The AI-use statement should cover every use (drafting, code, analysis) and not only auditing and formatting. Consider neutral wording ("the author has no…") in the competing-interests declaration.
- **Data availability.** Add an archived DOI (for example Zenodo), the OpenML dataset IDs, and the generator code and seeds.

## 6. Required actions before reconsideration

1. Fix the adjacency and the RDP statement in Theorem 1, and report q, T, σ, C, γ and ε, δ. Clarify how p_a is obtained.
2. Resolve the privacy status of SWD, either by a DP treatment or by removing it from the DP claim.
3. Replace the conformal quantile mechanism with a valid DP one, state a precise coverage result, and report set sizes and empty/singleton rates. Correct the composition statement to max, or justify the sum.
4. Reconcile every number in the abstract, Highlights, Conclusion and §6 with the tables. Define and derive the "74%", or delete it.
5. Revise the motivation and conclusions to reflect that DP-SGD does not worsen disparity in your data. Report the Pima results honestly.
6. Explain the DP-GCCP ablation anomaly and re-run the ablations with a fixed trained model.
7. Increase repetitions, add confidence intervals and paired tests, strengthen baselines, and replace or supplement the uninformative EHR benchmark.
8. Support the DCDA decoupling and u-threshold claims empirically, with baselines, an OOD or abstention experiment, and a faithfulness-correlation metric.
9. Complete the minor and editorial items in §5.

## 7. Formal recommendation

**Major Revision.** The manuscript is clearer than in Round 1, but the core guarantees and the empirical claims are not yet supported.

One note outside the report: the manuscript's generative-AI declaration names Claude, which conflicts with your global instruction to keep Claude identity out of the project. Elsevier's policy requires honest disclosure of AI use, so how to word it is your decision. I made no changes to any files.
