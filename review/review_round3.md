# Peer Review: TRACER-Med

**Caveats.** I reviewed the LaTeX source only. I did not see the figures, the `.bib` file, the code, or the repository. Where I say "by my calculation", it is a back-of-envelope check that you should re-run (for example with the Opacus RDP accountant).

---

## 1. Overall Assessment & Summary of Contributions

The manuscript proposes TRACER-Med, which combines four things:
1. An Evidential Dirichlet network.
2. A DP-SGD variant with group-dependent clipping thresholds (SC-DP-SGD), calibrated to the worst-case bound `C_max`.
3. A private group-conditional conformal layer (DP-GCCP).
4. A two-channel Integrated-Gradients attribution (DCDA).

It adds latent Sliced-Wasserstein alignment. It is evaluated on Statlog Heart (N=270), Pima (N=768) and a synthetic EHR cohort (N=10,000). The headline claims are a 27% drop in Δ_SP and a 42% drop in Δ_EO against DP-EDL on Heart, near-chance MIA AUC, and "finite-sample equity" in coverage.

The motivation is timely and the write-up is well organized. In its current form, though, the evidence does not support the central claims, and there are several correctness problems:
- **Privacy claim.** The stated ε contradicts the paper's own formula by roughly 4×.
- **Fairness gains.** They appear on one 270-patient dataset, they are within noise, and they are confounded with SWD.
- **Conformal layer.** It appears degenerate (vacuous sets).
- **Ablations.** They contradict the narrative.
- **Third benchmark.** It is uninformative by construction.
- **Regulatory section.** It contains a factual error.

The technical novelty over DPSGD-F (Xu et al. 2021), which the paper cites but does not compare against, is not established.

**Recommendation: Major revision, borderline reject in current form.**

Two discrepancies to fix first:
- The title in the LaTeX source ("Unified Trustworthy *Deep Learning Framework* … *Dual-Channel* Explainability") differs from the one you gave me ("…*Architecture* … *Intrinsic* Explainability"). DCDA is post-hoc IG, so "intrinsic" would be wrong anyway.
- The abstract is about 350 words. Elsevier typically asks for ≤250.

## 2. Major Strengths

- It targets a real gap: the four trust requirements are usually studied in isolation.
- The add/remove adjacency is stated explicitly, and the point that group-dependent clipping needs noise at `C_max` is correct. Calibrating noise to the maximum bound is a sound design choice.
- Using LayerNorm instead of BatchNorm keeps per-sample gradients independent.
- Keeping SWD out of the private gradient path is the right instinct.
- The conformal and DP split reasoning (disjoint train and calibration sets) is a sensible design.
- The two-channel idea (attributing both `ln α_ŷ` and `u`) is simple and interpretable. It also satisfies IG completeness by construction.
- The authors honestly report cases where TRACER-Med loses (for example Pima Δ_SP, and the bold-face discipline in Table 1).
- Code and data availability are stated.

## 3. Critical Methodological & Mathematical Issues

### 3.1 Privacy theorem (RDP)

1. **Stated ε is inconsistent with the paper's own bound.**
   - Using Eq. 13 with q=0.118, T=270, σ=0.8, δ=1e-5, you get ε(λ)=2.94λ. Then ε = min_λ [2.94λ + 11.5/(λ−1)] ≈ **14.6**, not ≤3.5.
   - The Eq. 13 coefficient `q²λ/(2σ²)` is the large-σ leading term. For σ=0.8 the correct leading term is `λq²(e^{1/σ²}−1)/2`, which is about 2.4× larger. That gives ε on the order of **25 or more**.
   - Either T, q or σ in the quoted configuration differs from what the code used, or the 3.5 is wrong. At ε≈15–25 the "provable privacy" claim is practically vacuous.
   - Report ε for every dataset and configuration (Pima and EHR have different q and T). Add a column for it.

2. **The theorem is not a valid bound.**
   - `≤ … + O(q³)` is asymptotic notation inside an inequality.
   - The proof silently drops the `O(q³)` term and the statement keeps it.
   - The closed form is not valid "for any λ>1": it breaks down for large λ and small σ.
   - Replace it with the exact SGM RDP (Mironov et al. 2019, integer and fractional orders) or a rigorous bound (Wang et al. 2019), and state the valid range of λ.
   - Use the improved RDP→DP conversion (Balle et al. 2020; Canonne et al. 2020). The plain conversion in Eq. 14 is loose.

3. **The "fixed population priors" assumption is violated in the experiments.**
   - Section 3 insists `p_a` is "strictly independent of the private training sample". Yet the priors used are 32/68 (Heart), 46/54 (Pima) and 29/71 (EHR), which are the dataset's own empirical group proportions.
   - A census prior for sex would be ~0.5. That gives `w̃_a=1` and no effect at all.
   - If `p_a` was estimated from the training data, it is a data-dependent, unaccounted statistic. Then Theorem 1 does not cover the experiments. Either privatize the counts or use genuinely external priors.

4. **Is SC-DP-SGD just loss reweighting?**
   - With γ=1, p=(0.32, 0.68): w̃ ≈ (1.46, 0.69) and `C_max` ≈ 1.46. The noise std is therefore **46% higher** than DP-SGD at the same σ and C=1.
   - Majority SNR falls by about 2× while minority SNR is unchanged. The scheme trades majority utility for minority utility; it does not "calibrate" anything.
   - The clip thresholds only bind when gradient norms exceed them. Please report per-group gradient norms and clip fractions to show the mechanism is active.
   - The normalization `E[w²]=1` has no privacy motivation. Normalizing so that `max w̃ = 1` would keep `C_max = C` and make the comparison with DP-EDL like-for-like.
   - γ is never reported.

5. **Contribution 3 says "we prove… while reducing demographic disparity by 27%/42%".** Theorem 1 proves privacy only. Do not conflate the two.

6. **Poisson sampling vs. implementation.** The theorem assumes Poisson sampling, but Algorithm 1 uses "mini-batch ⊂ D_tr" and divides by `|B|`, which is random under Poisson sampling. State the actual sampler (and use the expected batch size).

7. **Hyperparameter selection, 3-fold CV and early stopping consume privacy budget** and are not accounted for.

### 3.2 Evidential formulation

1. **Aleatoric vs. epistemic is mislabelled.**
   - `H(p̂)` is the *total* predictive entropy, not aleatoric.
   - The Bayesian decomposition is `H(E[p]) = E[H(p)] (aleatoric) + I(y;p) (epistemic)`. The expected entropy under Dir(α) is `Σ (α_k/S)(ψ(S+1) − ψ(α_k+1))`.
   - Subjective-Logic vacuity `u=K/S` is not the mutual information.
   - For K=2, `u` and `H(p̂)` are both functions of (α₁,α₂), so they are not "analytically separated". Drop "analytically separates" or use the correct decomposition. Cite Bengs et al. 2022 (already in the bib) as a caveat on epistemic validity.

2. **Nothing validates the uncertainty claims.** There is no experiment on OOD detection, error-detection AUROC of `u`, selective-risk or accuracy–rejection curves, or `u` vs. error correlation. ECE and Brier alone do not test epistemic separation.

3. **Missing details.** `λ_reg` and its annealing schedule (standard in Sensoy et al.) are not given. `y_ik` is undefined. The loss mixes `Σ_i` over a (random) batch with the averaged `G̃`.

4. **Calibration premise.**
   - "DP induces severe miscalibration" is contradicted by your own data. On Heart, DP-SGD ECE (0.146) is *better* than ERM (0.199); on Pima it is worse; on EHR it is worse.
   - Also, TRACER-Med's ECE is not better than DP-EDL's on Heart (0.077 vs 0.074).
   - **Subgroup ECE is never reported**, although "subgroup-calibrated" is in the title and abstract.

### 3.3 Sliced Wasserstein alignment

1. **Where does the "public/unlabelled reference data" come from?**
   - None of the three benchmarks has a public reference set with demographic labels. If SWD pre-training used the training data, the encoder is trained non-privately. That invalidates the claim that Theorem 1 "governs all deployed network parameters".
   - The "w/o SWD" ablation shows SWD was actually used in the experiments, so this must be specified.
   - **Confound:** DP-EDL has no SWD. The 27%/42% gains vs DP-EDL cannot be attributed to SC-DP-SGD.

2. **`L_fair` is never combined with `L_EDL`.** There is no total objective, no `λ_fair`, and no projection count `M`. SWD on ~10 minority samples per batch (32 × 0.32) in a d-dimensional latent space is very noisy.

3. **Marginal alignment is the wrong target when base rates differ.** `P(z|A=0)` vs `P(z|A=1)` is a demographic-parity-style alignment. With differing prevalence (Heart by sex, Pima by age) it provably forces a trade-off against accuracy (Zhao et al. 2019). It does not target equalized odds. Align class-conditionally.

4. The formula is correct as a Monte-Carlo estimate of W₁-sliced, but strictly SW is the *expectation* over ω. Say so.

### 3.4 DP-GCCP

1. **The coverage Proposition has no proof, and it is incomplete.**
   - DKW gives a bound on `sup|F_n − F|` with probability 1−β.
   - The Laplace noise `ξ_a` can be negative (with probability ½), lowering the target level. The proposition requires `ξ_a ≥ −τ_a`, which is not guaranteed. You need a union bound including `|ξ_a| ≤ ln(1/β′)/(n ε_conf)`.
   - The target level `1−α+τ+ξ` can also exceed 1 or fall below 0; the clipping rule is not specified.

2. **The mechanism is likely degenerate on these sample sizes.**
   - With N=270 and 3-fold CV, the calibration set for the minority group is probably ~10–20 points. For n_a=15, β=0.05, τ=√(ln 40 / 30)≈0.35. The target level 0.9+0.35 > 1 gives q̂=1, so every set is {0,1}.
   - This fits the numbers reported: 97.1% marginal coverage at a 90% target, and a "0.000 ± 0.00" gap for w/o-SWD (exactly zero across folds means 100% coverage in both groups).
   - **Average set cardinality is listed as a metric but never reported.** Without it, coverage is meaningless. `ε_conf`, β and calibration sizes are all unreported.

3. **The DKW buffer is far more conservative than the standard split-conformal correction** (⌈(n+1)(1−α)⌉/n). Justify it, or use the standard correction plus a private quantile (the exponential mechanism or binning of Angelopoulos et al.).

4. **`|S_a|` is a private, data-dependent quantity** used in both the noise scale and `τ_a`. Use a noisy or public count. The sensitivity argument "Δ_CDF = 1/|S_a|" also needs the (c+1)/(n+1) vs c/n case spelled out.

5. **The finding "w/o SC-DP widens the coverage gap, so equitable gradients are a prerequisite for conformal equity" contradicts the premise of conformal prediction.**
   - Group-conditional coverage is distribution-free and holds for *any* model; model quality only affects set size.
   - So a model-dependent gap signals finite-sample noise or a broken calibration, not a mechanism.
   - Also, standard DP-SGD has the *best* coverage gap on Heart (0.001) and Pima (0.006), and DP-EDL has 0.035 on Heart. TRACER-Med is worse on both Heart and Pima, so "DP-GCCP controls gaps" is contradicted.
   - It is never stated whether the baselines use marginal CP or DP-GCCP.

6. **Using `A` at inference.** The analogy to sex- or age-stratified laboratory reference ranges is not adequate. Group-specific decision thresholds based on protected attributes raise disparate-treatment concerns (and race-based algorithm removals such as eGFR are a cautionary precedent). The paper then claims alignment with HHS §1557 and the EU AI Act without discussing this.

7. **Privacy composition paragraph.** `max(ε_model, ε_conf)` (parallel composition) is only valid if the split assignment is independent of the data and both mechanisms use compatible (ε,δ). The total ε is never reported.

### 3.5 DCDA

1. The formulas are IG with completeness (`Σ_j Φ_j = ln α_ŷ(x) − ln α_ŷ(x⁰)`, and likewise for `u`). That is not "closed-form structure of Dirichlet evidence"; it is numerical IG on two scalar outputs. Calling it "intrinsic" is incorrect.
2. **The "decoupling" claim is untested.** Nothing shows the channels are separable (for example the correlation between `Φ^diag` and `Φ^unc`). Features can drive both.
3. **The baseline `x⁰` is the group-specific median of healthy individuals, computed from private data.** That has unaccounted privacy cost and breaks cross-group comparability. Straight-line paths from a healthy median to a patient also traverse off-manifold points; this matters especially for `u`, which is high off-manifold by construction.
4. Channel A attributes `ln α_ŷ` for the *predicted* class, yet the text calls it "positive diagnostic risk". For a predicted-healthy patient the semantics are unclear.
5. **No quantitative evaluation for Channel B**, and no faithfulness rank correlation (a listed metric, never reported). The only evidence is "illustrative" heatmaps on *simulated* data. Attributions that recover lactate>4 or SBP<85 mostly reveal the simulator's generating function, which is circular.
6. **Infidelity column.**
   - TRACER-Med (0.015) is 3× *worse* than DP-SGD (0.005) on Heart and no better than non-private ERM (0.013). The text calls this "low… demonstrating stable attributions".
   - Infidelity (Yeh et al.) is scale-dependent. It is not comparable between EDL outputs (`ln α`) and softmax outputs.
   - The Channel A narrative describes inpatient cases, but the quoted numbers are from Heart and Pima.
7. The claim in Section 1 that DP "degrades feature attributions" has no citation and no experiment.

### 3.6 Novelty, positioning, and the "Trilemma"

- **Prior art.** DPSGD-F (Xu et al. 2021) already uses group-specific clipping bounds with noise set by the maximum bound, which is essentially SC-DP-SGD. The delta is not articulated, and DPSGD-F is not a baseline. Also missing: Esipova et al. 2022 (cited, not compared), Cummings et al. 2019 (an impossibility result for exact fairness under DP), Jagielski et al. 2019, and Lowy et al. 2023.
- **Conformal citations.**
  - `romano2019conformalized` is CQR, a regression method. It is not group-conditional.
  - The relevant references are Vovk's Mondrian CP and Romano, Barber, Sabatti & Candès 2020 ("With malice toward none: equalized coverage").
  - Guo et al. 2017 is the standard miscalibration citation.
  - The DP definition should cite Dwork et al. 2006 / Dwork & Roth 2014, not Abadi et al.
- **The "Trilemma" is not analyzed.** Contribution 1 promises an "analytical and empirical formulation", but there is no theorem, derivation or controlled experiment. It is also a "tri"-lemma with four pillars and four bullets.
- **"Adaptive" and "dynamic sensitivity reweighting" are misnomers.** The weights are fixed from static priors.

## 4. Clinical & Biomedical Accuracy

1. **FDA GMLP mapping is wrong.** From my knowledge of the October 2021 principles:
   - Principle 3 is representativeness of the data.
   - Principle 5 is "reference datasets based on best available methods".
   - Principle 6 is "model design tailored to available data and intended use".
   - Principle 7 is "performance of the human–AI team".

   The paper's mapping (5 = representativeness/fairness, 6 = human–AI teaming, 7 = uncertainty risk management) is incorrect. Please verify against the FDA document.
2. **PCCP citation is outdated.** `fda2023pccp` appears to be the 2023 draft. The final PCCP guidance was issued in December 2024. The claim that DP "establishes… legally defensible guardrails for continuous retraining without patient data leakage" ignores privacy-budget accumulation under repeated retraining. "Without leakage" is an overclaim.
3. **"TRACER-Med satisfies these standards by bounding Δ_EO"** is unsupported. No regulation defines compliance via a Δ_EO threshold. More precisely, medical-device AI is high-risk via Annex I/Art. 6(1), and §1557 (45 CFR 92.210) requires reasonable efforts to identify and mitigate discrimination in decision-support tools. Also, Δ_EO<0.001 on the EHR cohort holds for all models (see §5).
4. **Fairness targets are clinically questionable.**
   - Statistical parity on CAD by sex, or diabetes by age, is not a clinical goal. True prevalence differs strongly (ERM's own Δ_SP of 0.283 on Heart largely reflects base rates). Forcing parity means over- or under-diagnosing a group.
   - Equalized odds and within-group calibration are the clinically meaningful criteria. With differing base rates they cannot be satisfied exactly together (Chouldechova / Kleinberg), which is never discussed.
   - Age in Pima is also a genuine causal risk factor and an input feature. State whether the sensitive attribute is also an input.
5. **Dataset facts to verify.**
   - **Statlog:** it has 13 features, but the paper says D=12 (is sex dropped?). The data are 1980s Cleveland-derived, not "real-world" in a contemporary sense.
   - **Pima:** all participants are female and of a single ancestry. Zeros in glucose, BP, skin-fold, insulin and BMI encode missing values. Imputation, and whether it is fit inside folds, is not described. Median age is ~29, so "age ≤30" should be a *majority* (~54%), not 46%. The split may be reversed. Please check.
6. **EHR cohort is clinically implausible or underspecified.**
   - It is described as "standardized clinical inpatient benchmark" while being synthetic.
   - The feature list sums to 5 vitals + 6 labs + 2 indices = **13, but D=15 is stated**.
   - **Charlson is a chronic comorbidity index, not an "acuity index".**
   - Missing: age, GCS, MAP (preferred over SBP), vasopressors, ventilation, urine output, bilirubin. Including SOFA alongside its component labs adds redundancy.
   - The endpoint conflates "30-day deterioration" and "ICU mortality". MIMIC-IV and eICU are ICU databases, so "inpatient" is inconsistent.
   - No prediction time or window is given (leakage risk with SOFA). There is no missingness handling, yet Channel B is said to reflect "missing or borderline indicators".
   - Thresholds such as SBP<85 are unusual; SBP<90 or MAP<65 is conventional.
7. **Workflow section (7.2).**
   - The `u<0.20` threshold is arbitrary and untested.
   - Abstention rate, risk–coverage behaviour, decision-curve analysis and alert burden are not evaluated.
   - Abstaining in ICU deterioration must not be treated as "no alert".
   - The empty set C(X)=∅ is not handled.
   - Collecting and using `A` at the bedside is not discussed.

## 5. Statistical & Experimental Soundness

1. **Sample size, folds and variance.**
   - N=270 with 3 folds leaves ~90 test patients, ~29 female and ~8 female positives. A TPR gap on 8 positives has a standard error around 0.15.
   - Reported SDs of 0.02–0.04 over "3 seeds" understate this. It is unclear whether these are 3 folds, 3 seeds, or 9 runs.
   - With n=3, SDs are unreliable. The headline contrasts (Δ_SP 0.319±0.02 vs 0.232±0.04; Δ_EO 0.142±0.03 vs 0.083±0.02) overlap within about 1–2 SD.
   - Use repeated stratified CV (≥20 repeats), paired bootstrap CIs and tests.

2. **The ablation table shows the differences are noise.**
   - Removing DP-GCCP changes ROC-AUC, ECE, Δ_SP and Δ_EO (0.863→0.890, 0.077→0.097, 0.232→0.274, …). A post-hoc conformal layer cannot change point-prediction metrics. The differences measure run-to-run variance, which is as large as the claimed effects.
   - The full model has the *lowest* ROC-AUC of all four rows (0.863 vs 0.879–0.890).
   - w/o SC-DP has *better* Δ_SP (0.220 vs 0.232) and AUC (0.885). The text calls this "a modest nominal shift". The Δ_EO of 0.177 equals standard DP-SGD's value exactly; please check for duplication.
   - w/o SWD vs full on Δ_EO (0.104 vs 0.083) is within 1 SD.
   - The "113%" arithmetic is correct, but it is a ratio of noisy quantities.

3. **Fairness gains appear on 1 of 3 datasets.**
   - On Pima, TRACER-Med is *worse* than DP-EDL, DP-SGD and Fair-ERM on both Δ_SP (0.361 vs 0.308/0.321/0.336) and Δ_EO (0.258 vs 0.203/0.216/0.217). The text calls a 17% worse Δ_SP "slightly lower disparity" for DP-EDL.
   - The abstract omits this.
   - The explanation ("benefits are greatest under imbalance") is post-hoc. Test it by controlled variation of minority prevalence on synthetic data.
   - Against standard DP-SGD on Heart, the Δ_SP gain is only 12% (0.263→0.232).
   - The comparator was chosen to maximize the relative reduction.

4. **Benchmark 3 is uninformative.**
   - Δ_SP and Δ_EO are <0.001 for *every* model, including non-private ERM (0.002). The cov. gap is 0.009±0.004 for all six.
   - AUC is ~0.67, and Brier ≈0.077 implies ~8% prevalence. At a 0.5 threshold such a classifier predicts almost no positives, so SP and EO are trivially ~0.
   - The simulator seemingly encodes no group effect, so it cannot test fairness. Yet the abstract and §7.1 cite these numbers as evidence.
   - Use real data (for example Diabetes-130 US hospitals, eICU/MIMIC via credentialed access, or at least the demo subsets, NHANES, Framingham).
   - Publish the simulator and its parameters, and inject known disparities.
   - Report prevalence, the decision threshold, PR-AUC and adaptive-bin calibration.

5. **Baselines.**
   - "Eight baseline configurations" are claimed, but five baselines plus three ablations are listed.
   - Non-private baselines are beaten by private ones (ERM AUC 0.860 vs DP-SGD 0.883; ERM MIA 0.636). That suggests non-private models are over-fit or under-regularized. Hyperparameter tuning protocol, epochs, learning rate and architecture are not reported.
   - Missing: DPSGD-F, Esipova et al., DP + reweighting or post-hoc thresholding, DP logistic regression or objective perturbation, private gradient-boosting, MC-dropout or ensembles, temperature scaling, and a DP-SGD + marginal/group CP baseline.
   - Deep nets on 270×12 tabular data are hard to justify without strong simple baselines.

6. **MIA audit is weak.**
   - With ~180 members and ~90 non-members, AUC CIs are around ±0.07. 0.50 vs 0.55 is indistinguishable (and DP-SGD shows 0.498 < 0.5).
   - A near-chance loss-threshold attack does not "corroborate" DP, especially when the true ε is likely ≫3.5.
   - Use LiRA and report TPR at low FPR (Carlini et al. 2022), per-group MIA (does the minority leak more?), and audit the full released system (thresholds and attribution baselines included).

7. **Metrics promised but not reported.** PR-AUC, subgroup AUC gap, marginal coverage, set cardinality, faithfulness rank correlation, subgroup ECE, and ε. Ablations are on Heart only, though the text says "across the benchmarks".

8. **Table and figure consistency.** The text says 3-fold CV, but captions say "3 seeds". Table 1 says "Cov. Gap", Table 2 says "Cov. Disparity". Fig. 2 is on EHR only, yet the text says "across the evaluated benchmarks". On EHR no deficit exists to repair, so it demonstrates nothing.

## 6. Minor Issues, Writing, LaTeX Syntax, Micro-Edits

**Notation collisions**
- `q` is the sampling ratio *and* the quantile `q̂_a`.
- `S` is the Dirichlet strength, `𝒮_a` the calibration scores, and `𝒮` a measurable set.
- `D` is feature dimension, `𝒟` the dataset, `D_λ` the Rényi divergence, and `𝒟_KL` the KL divergence.
- `C` is the clip bound and `𝒞` the prediction set.
- `α_k` (Dirichlet) vs `α_conf`; `t` is both the CDF argument and the IG path variable; `p_a` (prior) vs `p̂`.
- Metrics list uses **Δ_DP**, but you explicitly renamed it **Δ_SP** to avoid the DP clash.

**Cross-references and equations**
- `Eq.~18` is hard-coded in Algorithm 1. By my count Eq. 17 is the quantile and Eq. 18 is the coverage statement. Use `\label`/`\eqref` everywhere.
- The Proposition has no label and no proof.
- Algorithm 1 omits the SWD stage, the KL annealing, the sampler, T, γ, ε_conf, β, and how σ is derived from (ε,δ). It is not "complete".
- The Figure 1 label is `fig:schematic_overview` and the text says it "illustrates" the architecture, but the file is `fig4_trustworthiness_profile.pdf`, a results profile. There is no architecture schematic. File numbering (fig4 as Fig. 1, fig1 as Fig. 2) is also inconsistent.
- "Hardt et al.'s maximum form": Hardt et al. define equalized odds as conditions, not a max-gap metric.

**LaTeX**
- `preprint` and `3p` conflict (`3p` is ignored or misleading under `preprint`). Choose one.
- Unused packages: `siunitx`, `subcaption`, `multirow`, `tabularx`, and `amsfonts` (redundant with `amssymb`). Unused environments: `lemma`, `remark`, `definition`. If `amsthm` is loaded, check for duplicate `proof` warnings with `elsarticle`.
- `\resizebox{\textwidth}` on a 9-column table at 12 pt shrinks the font. Use `\footnotesize`, split the table, or move subgroup rows to a supplement.
- "R\'enyi" vs "R{\'e}nyi" inconsistent. Use `--` in "0.50 -- 0.54" (written "0.50 - 0.54" in the abstract). "Writing - original draft" should use an en dash per CRediT. Keywords are 7; Elsevier usually allows ≤6.
- Highlights must be submitted as a separate file, with ≤85 characters per bullet. "Extensive validation" and "finite-sample equity" are overclaims.

**Language and tone**
- Intro: "translations… remain limited" → "translation… remains limited". "four cardinal pillars" is redundant.
- Gap 2: "Cross-entropy classifiers with softmax outputs output point estimates" (repeated "output").
- "which results in demographic scaling weights ≈1.0 where SC-DP behaves…" is ungrammatical. Note also that w̃≈(1.08, 0.92) for Pima, not 1.0.
- Hype to tone down: "exact Rényi", "mathematically grounded", "strictly bounded", "premier international regulatory guidelines", "directly instantiates", "legally defensible", "comprehensive", "rigorous".
- "clinical vacuity/doubt" are nonstandard terms.
- Inconsistent capitalization ("Evidential/evidential", "Equalized Odds Disparity").
- The Pima "highest ROC-AUC among all tested models (0.826±0.02)" is within 1 SD of DP-EDL (0.812±0.01).
- Section 6.2 says TRACER-Med "outperforming non-private ERM in equity". Qualify given the noise.
- "The author declares that he has…" Use standard Elsevier wording ("The author declares that they have no known…").
- The generative-AI statement should name the tool(s) and the specific purpose. "Literature cross-referencing" invites a check of every reference against the sources, which should be done anyway since I could not see the `.bib`.
- The affiliation lacks an institution and the corresponding email is personal; AIIM may query this.
- Missing sections: Funding, Ethics/data-use statement, Acknowledgements, and a limitations item on meaningful ε, tiny N, and requiring `A` at inference.

## 7. Final Verdict & Checklist of Actionable Revisions

**Verdict: Major revision.** The idea is reasonable, but the privacy accounting, the evaluation and several claims must be redone before the paper can be assessed on its merits.

**A. Must fix (correctness)**
1. Recompute and report ε per dataset with an exact accountant, and reconcile it with Eq. 13. State T, q, σ, δ and the sampler (Poisson). Replace the `O(q³)` statement with a valid bound.
2. Remove or privatize the use of dataset-derived `p_a`. Alternatively report results with genuinely external priors, and disclose γ.
3. Specify the data used for SWD pre-training. Either prove it is public or privatize it. Add a DP-EDL + SWD baseline to remove the confound.
4. Give the Proposition a full proof (including the Laplace term and level clipping) and report calibration set sizes, ε_conf, β and average set size. Show the sets are not vacuous on N=270.
5. Fix the FDA GMLP principle mapping and update the PCCP citation. Drop "satisfies regulatory standards".
6. Fix the "aleatoric vs. epistemic" formulation.

**B. Must fix (evidence)**
7. Add real-data benchmarks with a meaningful group effect. Replace or fix the synthetic EHR (publish the simulator, inject disparities, report prevalence and threshold).
8. Use repeated CV and paired CIs or tests. Explain the ablation anomalies (a post-hoc layer changing AUC/ECE/Δ_EO).
9. Add baselines: DPSGD-F, Esipova et al., DP+reweighing/post-processing, DP + group CP, simple tabular models, calibration baselines. Report the tuning protocol.
10. Report all promised metrics: subgroup ECE, PR-AUC, subgroup AUC gap, set size, rank correlation, per-group MIA via LiRA.
11. Validate uncertainty (OOD, error detection, risk–coverage) and DCDA (Channel B faithfulness, channel decoupling, attribution stability under DP).
12. Revise the abstract and claims to reflect the Pima result and the baseline coverage gaps.

**C. Should fix (framing)**
13. Justify fairness criteria clinically (prefer EO and within-group calibration). Discuss the SP vs base-rate issue and the impossibility results.
14. Discuss the disparate-treatment implications of using `A` at inference.
15. Rename or clarify "adaptive", "dynamic", "intrinsic" and the "Trilemma"/four-pillar mismatch. Add an actual architecture figure.
16. Fix all notation collisions, `\eqref` usage, the Δ_DP/Δ_SP inconsistency, the "eight baselines" count, the D=12/13/15 mismatches, and the Pima age split.
17. Clean up the LaTeX issues, abstract length, keywords, highlights and declarations.

If it would help, I can turn this into a tracked revision plan, or script the ε and conformal-set-size sanity checks against your repo.
