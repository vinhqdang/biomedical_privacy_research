# Peer review: TRACER-Med (Artificial Intelligence in Medicine)

I reviewed the LaTeX source you pasted. I did not open the `.bib`, the code, the data or the compiled figures. Figures 1–4 and the reference list are therefore unchecked. Every number below comes from the manuscript, and the clipping-weight arithmetic is my own calculation from its stated priors.

## 1. Overall assessment and summary of contributions

**Claims.** The manuscript proposes a five-part pipeline:
- Dirichlet evidential learning.
- Group-adaptive clipping in DP-SGD (REF-DP).
- A class-prototype alignment loss (PC-PAT).
- A histogram-based private group-conditional conformal procedure (DP-GCCP).
- Integrated-gradients attribution applied to both the disease probability and the epistemic mutual information (DCDA).

It evaluates the pipeline on Statlog Heart (N=270), Pima (N=768) and a simulated EHR cohort (N=10,000).

**Verdict: major revision, bordering on reject in current form.** The framing is timely and the writing is fluent. The central technical claims do not survive scrutiny, and the empirical evidence does not support the headline narrative.

1. **PC-PAT privacy claim.** Its "strict per-sample independence" is contradicted by Algorithm 1, line 2. The prototypes are computed from the whole training set.
2. **DP-GCCP theorem.** It is incorrect as stated, for four separate reasons (see §3).
3. **Heuristics outside the theory.** The experiments rely on a 0.96 cap and shrinkage toward a marginal threshold. These void the PAC guarantee and are not covered by the privacy proof.
4. **Results.** TRACER-Med does not beat DP-EDL on most metrics. The ablation contradicts the module-level claims. All differences are within noise on 90-patient test folds.

The novelty is also thinner than presented. Theorem 1 is standard DP-SGD with heterogeneous clip norms. DCDA is plain Integrated Gradients applied to two scalar functions. The "Riemannian" and "information-geometric" language is branding rather than method.

## 2. Major strengths

- The problem is well-motivated. The interplay of DP, subgroup fairness, UQ and explanation in clinical ML is real.
- The paper is transparent that Statistical Parity and Equalized Odds are incompatible under unequal base rates. It also flags vacuous prediction sets (the 0.000 coverage gap) as a deceptive result. That is a useful and honest observation.
- Add/remove adjacency, Poisson subsampling and an RDP accountant are stated explicitly. Per-sample LayerNorm is a correct choice for DP.
- The Dirichlet entropy, mutual-information and KL formulas are correct (verified in §3).
- The reference baselines for DCDA are an improvement over arbitrary medians. The regulatory section is mostly accurate on citations.
- Code, data sources and OpenML IDs (53, 37) are disclosed. The vacuity-collapse footnote is candid.

## 3. Critical methodological and mathematical issues

### 3.1 Verified correct

- Expected Dirichlet entropy and the mutual information expression.
- The Bayes loss and the KL to Dir(1) with $\tilde\alpha=\mathbf y+(1-\mathbf y)\odot\alpha$.
- The gradient $\partial\mathcal L/\partial z_j=(\psi_1(S)-y_j\psi_1(\alpha_j))\sigma(z_j)$.
- The Dirichlet FIM $\psi_1(\alpha_j)\delta_{jk}-\psi_1(S)$ and its pull-back $J^\top G J$.
- The trigamma expansion and $\psi_1(1)=\pi^2/6$.
- DKW with $\sqrt{\ln(4/\beta)/2n}$ giving $\beta/2$.
- The histogram $\ell_1$ sensitivity of 1 and the Laplace mechanism.
- The Bernstein form used for Laplace sums.
- The conversion $\epsilon(\lambda)+\ln(1/\delta)/(\lambda-1)$.

### 3.2 Errors and gaps

**[Critical] PC-PAT is not per-sample (Theorem 1 and Highlight 3).**
- Algorithm 1 line 2 sets $\bar\mu_y^*$ to the class mean of $h_\theta(x_i)$ over all of $\mathcal D_{tr}$. Every $g_{j,\text{fair}}$ therefore depends on every other record.
- Adding or removing $r^*$ changes $\bar\mu_y$ and hence all clipped gradients in the batch. Per-step sensitivity is no longer $C_{\max}$.
- The proof's sentence "the gradient $g_j$ of any record $r_j\ne r^*$ is independent of $r^*$" is false for this implementation.
- The text does not say whether the prototypes are refreshed during training. Either way they are an unaccounted data-dependent release.
- Fix options: public or prior prototypes, learnable prototypes treated as parameters (the gradient is then per-sample), or privately estimated prototypes with a budget.

**[Critical] Theorem 2 depends on non-private $n_a$.**
- $\tau_a$ uses the true cohort size $n_a$. So do the shrinkage weight $\lambda_a=n_a/(n_a+15)$ and the cap logic.
- Adding or removing a record changes $n_a$, so $\hat q_a$ is not post-processing of the noisy histogram. The $(\epsilon_{conf},0)$-DP claim fails.
- The marginal threshold $\hat q_{\text{marg}}$ in the shrinkage step is a second release with no stated budget.
- Fix: replace $n_a$ with $\tilde N_a$ plus a high-probability bound on its error, or spend budget on $n_a$.

**[Critical] The theorem is false with the 0.96 cap.**
- The definition of $\hat q_a$ includes $\min(1-\alpha+\tau_a,0.96)$. The proof needs $\tilde F(\hat q)\ge1-\alpha+\tau_a$.
- Whenever $\tau_a>0.06$, the conclusion $F_a(\hat q_a)\ge1-\alpha$ does not follow.
- With the manuscript's constants ($M=20$, $\epsilon_{conf}=1$, $\beta=0.05$): at $n_a=50$, the DKW term is 0.21 and the Laplace term is 0.64, so $\tau_a\approx0.85$. Even at $n_a=1000$, $\tau_a\approx0.08$.
- The cap therefore binds on Heart and Pima, and probably on the EHR minority cohort.
- The reported coverage is thus an empirical heuristic, not a PAC guarantee.

**[Major] The Laplace concentration step has the wrong constant.**
- Solving the stated Bernstein bound with $V=2M/\epsilon^2$ and $b=1/\epsilon$ gives $t\le 2\sqrt{M L}/\epsilon+2L/\epsilon$ with $L=\ln(4M/\beta)$.
- The manuscript has $\sqrt{2ML}/\epsilon$. The leading term is off by $\sqrt2$.

**[Major] The proof treats $\tilde F_a=\hat F_a+\text{noise}/n_a$.**
- The actual quantity is a ratio of noisy cumulative counts to a noisy, truncated total.
- The proof ignores the denominator error and the $\max(0,\cdot)$ bias. Each empty bin contributes $E[\max(0,\mathrm{Lap}(1))]=\tfrac12$ spurious count.
- With 20 bins that is about 10 phantom counts, a large share of a cohort of roughly 15–30 patients.

**[Major] No end-to-end privacy statement.**
- The model is trained at $(\epsilon_{train},\delta)$ and the calibration releases cost $\epsilon_{conf}$.
- If $\mathcal D_{cal}\cap\mathcal D_{tr}=\emptyset$, the total is a max over disjoint data. If not, it is a sum. The text never says which.
- The train/calibration/test split sizes are never stated.
- The preprocessing statistics (means, standard deviations, imputation, Pima zero-handling) are computed on training folds without DP.
- The tuning of $0.96$, $15$, $\lambda_{fair}$, $C_0$ and the epochs is not covered. The paper cites Papernot and Steinke and then ignores them.

**[Major] REF-DP is not "Riemannian" and its weights are ad hoc.**
- Clipping with the Euclidean norm at group-dependent thresholds is not Riemannian clipping. A Riemannian norm would be $\|g\|_{G^{-1}}$.
- The Dirichlet FIM is the metric on the distribution over $p$. It is not the Fisher information of the loss gradients or of the predictive distribution. The leap from $\operatorname{Tr}G$ to "gradient variance" is unsupported.
- "Hyperbolic curvature" conflates the manifold's sectional curvature with the magnitude of $\operatorname{Tr}G$.
- $\bar{\mathcal F}_a\approx K(K-1)/\bar S_a$ holds only for uniform $\alpha$ and replaces $E[1/S]$ with $1/E[S]$. It is poor near $\alpha\to1$, the regime the argument relies on ($\psi_1(1)=1.645$, not 1).
- The factor $(1/2p_a)^{1/2}$ is unmotivated. It is "1/2" for two groups only by coincidence.

**[Critical] The "epidemiological priors" are not external.**
- $p_0=0.678,\ p_1=0.322$ are exactly $183/270$ and $87/270$ of this dataset. Pima's $0.544/0.456$ are also the sample proportions.
- The "prior expected evidence" values (3.2/2.4, 2.8/3.0, 3.5/2.5) are not epidemiological quantities. They have no source or derivation and sit near the initialization value $S\approx3.4$.
- If estimated from data, they leak privacy. If invented, the method is tuned by hand.

**[Critical] The "without expanding global sensitivity" claim is false, and REF-DP's benefit is mis-stated.**
- From the paper's own priors, I compute $\tilde w=(0.80,1.33)$ on Heart and $(0.77,1.39)$ on EHR.
- So $C_{\max}\approx1.33$–$1.39\,C_0$, versus $C=1$ for the DP-SGD baseline. The noise std rises 33–39% while the majority clip falls to roughly 0.8.
- On Pima, $\tilde w\approx(0.97,1.03)$. The REF-DP clipping there is essentially uniform, so Pima's EO gain cannot come from REF-DP.
- "Selectively amplifies minority SNR" holds only against a $C_{\max}$-uniform baseline, which is never run.

**[Major] The PC-PAT theory is loose.**
- Minimizing $\mathbb E\|h-\mu\|^2$ gives $W_2(P_{a,y},\delta_\mu)^2$. By the triangle inequality $W_2(P_{0,y},P_{1,y})$ is then bounded above, and it reaches zero only when the loss does.
- "Monge–Kantorovich duality" does not give "directly minimizes to zero".
- PC-PAT is a center loss that pulls everything toward the class mean regardless of group.

**[Major] The RDP expansion at $\sigma_{DP}=0.8$.**
- The displayed $\epsilon(\lambda)\approx Tq^2\lambda/(2\sigma^2)+O(\cdot)$ is a large-$\sigma$, small-$q$ approximation. At $\sigma<1$ it is not valid. State the exact accountant instead of presenting it as a bound.
- Composition is adaptive, not "independent".
- Heart $\epsilon=15.36$ is not a meaningful privacy level. The Related Work and regulatory sections should not imply otherwise.

**[Minor] Statement-level errors.**
- Vacuity $u=K/S$ lies in $(0,1]$, not $(0,1)$. It equals 1 at zero evidence.
- The $\ell_2$-sensitivity statement "difference is $0$ or $\pm\tilde g^*$" is stated without addressing subsampling amplification.
- Bin $(t_0,t_1]$ excludes $s=0$. Use $[t_0,t_1]$.
- DP-GCCP needs the sensitive attribute at inference time. That is a clinical and legal limitation never discussed.

**[Major] DCDA.**
- It does not use Dirichlet geometry. It is IG on $\hat p_1$ and on $\mathcal I$. The "exploiting Dirichlet information geometry" claim is overstated.
- Completeness holds only for the exact integral. Report the Gauss–Legendre residual.
- The claim that a standardized baseline "prevents off-manifold interpolation across categorical variables" is wrong. A straight line to any single baseline still passes through fractional categories.
- The Statlog `fasting blood sugar` is a binary flag (>120 mg/dL). Setting it to 90 mg/dL is off-manifold.
- Pima glucose is a 2-hour OGTT value, not fasting.
- Statlog `chest pain = 4` is "asymptomatic". It is the category most associated with disease, so "clinically asymptomatic reference level" is ambiguous or inverted.
- Pregnancies = 0 and age have no physiological "healthy" value.
- There are no sanity checks (randomization or deletion tests) and no comparison with SHAP or IG using alternative baselines.
- Prior work on uncertainty attribution (CLUE and path-attribution of epistemic uncertainty) is cited but not contrasted.

**[Major] EDL itself.**
- The literature on EDL's UQ being unreliable is not engaged. Examples are Bengs et al. 2022 (cited), plus Shen et al. 2024 ("Are UQ capabilities of EDL a mirage?") and Jürgens et al. 2024 (ICML).
- The workflow thresholds on vacuity $u$, while Channel B attributes the mutual information. These are different epistemic measures.

**[Major] The "Trilemma" is not demonstrated.**
- The three trade-offs are asserted with citations but never measured. Trade-off 2 ("clipping is an evidence suppressor") has no evidence.
- The intro's $\tau_a\propto 1/\sqrt{n_a}+1/(n_a\epsilon)$ contradicts the theorem's $\sqrt{M}/(n_a\epsilon)$ terms.

## 4. Clinical and biomedical accuracy

**Statlog Heart**
- The prevalence figures are correct (about 52% in males and 29% in females).
- `ca` (vessels coloured by fluoroscopy) is an angiographic finding. Using it to "predict" angiographic CAD is circular and leaks the label. `thal` is also an invasive/stress-test result. The model is not a realistic screening tool.
- Sex is correctly excluded from $D=12$. Age is a legitimate risk factor but is also the Pima protected attribute, so $A$ is a deterministic function of an input there.

**Pima**
- Prevalence differs strongly by age ($\sim$20% under 30 versus $\sim$50% above). Large $\Delta_{SP}$ values are therefore partly legitimate.
- The $\le 30$ vs $>30$ split is not a protected-class analogue of sex or race.
- The cohort is a single Indigenous community of women. Add an ethics discussion on its provenance.
- How zeros are imputed (about 49% of insulin values and 30% of skin-fold values are "missing") is not given.

**Simulated EHR**
- Outcome prevalence of about 42% for "emergency ICU transfer or 30-day mortality" is far above published ward-deterioration rates (roughly 5–15%).
- SOFA is a derived score that includes MAP, creatinine, bilirubin, platelets and PaO₂/FiO₂. Including it alongside those inputs creates deterministic collinearity. SOFA is also an ordinal integer, not a "continuous" feature.
- MAP and SBP are nearly collinear.
- Lactate, ABG and bilirubin are missing not-at-random in real ICUs. The prediction time versus the outcome window is undefined.
- The minority cohort and its "access disparities" are undefined. The simulator's generative mechanism, correlations and disparity injection are undisclosed.
- Fairness findings on simulated data reflect what the authors injected. Qualitative DCDA findings (lactate, MAP, SOFA as drivers) are circular if the simulator uses them to generate the outcome.
- Citing "MIMIC-IV and eICU" distributions raises a DUA (data use agreement) question. Confirm that derived parameters are allowed.

**Workflow section**
- $u=K/S$ with $K=2$ means $u<0.20$ requires $S>10$. The priors have $\bar S\approx2.4$–$3.5$, i.e. $u\approx0.6$–$0.8$.
- Combined with set sizes of 1.80–1.87, the proposed rule would abstain on roughly 87% or more of patients. No abstention rate, selective-risk curve or clinician study is reported.
- Sensitivity is typically prioritized in screening. There is no decision-curve analysis and the decision threshold is a fixed 0.5.
- The $u$ cut-off is arbitrary.

**Regulatory section**
- Principles 3 and 10 are over-mapped. GMLP Principle 3 concerns data collection, which REF-DP cannot ensure. Principle 10 concerns monitoring and drift, not privacy bounds.
- DP budget is consumed by every retraining round. The PCCP paragraph ignores this.
- $\Delta_{SP}$ is not an "error disparity". Reporting $\Delta_{SP}$ for Heart and $\Delta_{EO}$ for Pima is selective.
- The citations (FDA GMLP 2021, PCCP final guidance December 2024, EU AI Act Articles 10/13/14/15, 45 CFR §92.210) are accurate. Verify current EU AI Act high-risk application dates and mention Articles 9 and 12.
- ε=15 is not anonymization under HIPAA or GDPR.

## 5. Statistical and experimental soundness

**Evidence versus claims**

| Dataset | TRACER-Med result | Reading |
|---|---|---|
| Heart | $\Delta_{SP}$ 0.242 is best. DP-EDL is better on ROC-AUC, ECE, Brier, $\Delta_{EO}$ (0.115 vs 0.158) and MIA (0.542 vs 0.565) | The claimed win is on one metric the paper itself deprioritizes |
| Pima | Best on $\Delta_{EO}$, $\Delta_{SP}$, ECE, Brier, MIA | Margins of 0.002–0.003 on ECE, Brier and MIA are meaningless. REF-DP is ~uniform here |
| EHR | $\Delta_{SP}$ 0.021 (DP-EDL 0.010), coverage gap 0.028 (DP-SGD 0.003, EDL 0.007, DP-EDL 0.015), infidelity 0.010 (DP-EDL 0.006), set size 1.83 (DP-EDL 1.81) | No demonstrated benefit over DP-EDL on the one adequately sized dataset |

**Other issues**

1. **Baseline strength.**
   - ERM AUC is 0.845 on Heart and 0.781 on Pima. These are below typical regularized logistic regression results (~0.90 and ~0.83).
   - DP-SGD beating non-private ERM by about 3 AUC points is a sign of an under-regularized baseline, not "implicit regularization".
   - Missing baselines: logistic regression, gradient boosting, DP logistic regression/objective perturbation, DPSGD-F (Xu), Lagrangian DP-fair (Tran), gradient-misalignment methods (Esipova), Lowy et al. 2023, and a non-private fair EDL.
2. **Unfair conformal comparison.**
   - Baselines appear to get the raw uncapped $\tau_a$ and collapse to 2.00. TRACER-Med gets the cap and shrinkage.
   - "Vacuity collapse" is therefore an artefact of the authors' own calibration procedure, not a property of DP models. Apply the identical conformal procedure to every model.
   - The vacuous results are also bolded as "best" in Table 1.
3. **Conformal coverage is never shown as coverage.**
   - Marginal coverage is not reported. With a set size of 1.87 on a binary task, only about 13% of sets are singletons, so coverage is likely about 0.99, far above the 0.90 target.
   - Class-conditional coverage is also missing. $1-\hat p(y|x)$ can hide poor coverage for the positive class.
   - "Informative, non-vacuous" is an overstatement.
4. **Noise floor of the metrics.**
   - A Heart test fold has 90 patients and about 29 women, roughly 8 of them diseased. One patient moves TPR by about 0.12.
   - The standard error of group coverage is about 0.056, so a gap of 0.024 is far inside the noise.
   - $|{\rm Cov}_0-{\rm Cov}_1|$ is positively biased.
   - ECE with 10 bins on 90 samples and MIA AUC on tiny train/test sets (SE ≈ 0.05–0.07) are not interpretable at the reported precision.
5. **Statistics.**
   - There are 3 folds and no CIs, paired tests or multiple-comparison control. Most "reductions" lie within one SD (e.g. 0.242±0.02 vs 0.296±0.03, or EO 0.191±0.01 vs 0.235±0.08).
   - The SD for TRACER-Med on Pima ($\pm0.01$) looks implausibly small beside its neighbours.
   - Whether the std is over folds or seeds is ambiguous ("3-fold CV with seeds 42, 43, 44").
6. **Ablation (Table 2) contradicts the narrative.**
   - Every ablation improves $\Delta_{EO}$: 0.158 → 0.132, 0.123 and 0.118. EO is the paper's declared primary criterion, and PC-PAT is supposed to target it.
   - Removing PC-PAT also improves AUC, ECE and coverage disparity.
   - Removing DP-GCCP (marginal CP) gives a *lower* coverage disparity (0.008 vs 0.024). That is the opposite of its purpose.
   - Marginal CP cannot alter AUC, ECE, SP or EO, yet those values change (SP 0.261, EO 0.118). This indicates retraining noise of ±0.02–0.04. All ablation differences are therefore uninterpretable.
   - "w/o SC-DP" reports SP exactly 0.318±0.03, identical to Standard DP-SGD. That is suspicious and should be checked for a copy error.
   - Missing ablations: cap/shrinkage (the heuristics that do the work), EDL vs softmax, PAT vs SWD, and a $C_{\max}$-uniform clipping baseline.
   - The text says "across the benchmarks", but the table covers Heart only.
7. **Likely trivial coverage gaps.** With $N=270$ and 3-fold CV, minority calibration cohorts may be only about 12–17 patients. Shrinkage to the marginal threshold ($\lambda_a\approx0.5$) then makes group thresholds approximate the marginal one. The small gap would then be by construction, consistent with the ablation.
8. **Privacy auditing.** MIA is not distinguishable between TRACER-Med and DP-EDL/DP-SGD on EHR (0.505, 0.506, 0.508). On Heart, TRACER-Med is higher (0.565 vs 0.548 and 0.542). Report ε-sweeps, privacy–utility curves and a canary-based empirical ε lower bound.
9. **Calibration.** ECE on EHR is 0.145 for private models versus 0.028–0.043 non-private. A title claiming "uncertainty-calibrated" is not supported. Report reliability diagrams.
10. **Misstated results in the text.**
    - §6.2 says DP-SGD "degrades calibration, ECE to 0.174 on Heart". ERM's ECE on Heart is 0.201, so DP-SGD actually improves it.
    - "All private models … ROC-AUC 0.746–0.748 and Brier 0.225–0.226" omits DP-SGD (0.743, Brier 0.254).
    - The 13% and 20% singleton rates are quoted in text but not in Table 1.
    - The Heart "Δ_SP matches the base-rate gap" argument is post hoc and conflicts with using lower Δ_SP as a win elsewhere.
    - The radar-chart claim that TRACER-Med "encloses" the baselines is contradicted by Table 1 (DP-EDL dominates it on Heart in five of six metrics). Radars on $1-\text{metric}$ axes also compress differences.
    - The abstract says "baseline DP models collapse to size 2.00". That occurs only for DP-EDL on Heart and DP-SGD on Pima.
    - Infidelity: low values for DP models mostly reflect smaller gradient scale. The text calls this "stable attributions". Infidelity is not stability, and Channel B infidelity is not reported. On EHR, TRACER-Med's 0.010 is worse than ERM (0.002) and DP-EDL (0.006).
11. **Reproducibility and tuning.**
    - Missing hyperparameters: $\lambda_{fair}$, latent dimension, shrinkage constant provenance, bin edges, calibration split, stratification variable.
    - "About 4,200 parameters" does not match my count of roughly 2.9k–3.4k for D=8, 12 and 15 with 64/32 layers plus LayerNorm.
    - There is no statement that tuning was kept off the evaluation folds.

## 6. Minor issues, writing, LaTeX and micro-edits

**Front matter**
- The title in the source differs from the title in your request. The source omits "Dual-Channel Explainability", though DCDA is one of five pillars.
- Highlights are limited to 85 characters. Four of the five exceed it (roughly 89–92 characters). Highlight 3 asserts the flawed privacy proof.
- The abstract is about 330 words and number-dense. Many Elsevier journals cap it near 250. It mixes headline metrics (SP on Heart, EO on Pima) selectively.
- The abstract says "guaranteeing strict per-sample record independence" and "mathematically grounded". Both overclaim (see §3).

**Notation**
- σ is both the sigmoid and σ_DP. λ is the RDP order, λ_a (shrinkage) and λ_fair. β is β_e (annealing) and β_conf. q is the subsampling rate and $\hat q_a$ the conformal quantile. $t$ is the grid point, the path parameter and the Bernstein threshold. $e$ is the epoch, $e_k$ the evidence and Euler's $e$. $S$ is evidence and $\mathcal S_a$ the calibration partition. $C$ is the clip and $\mathcal C$ the prediction set. $\epsilon(\lambda)$ and $\epsilon_{conf}$ are both used.
- The notation list is missing $K$, $d$, $C_0$, $T$, $q$, $\lambda_{fair}$, $N_{tr}$ and others, and lists implementation constants like $\nu$ instead.
- The sentence "$\nu$ … distinguished from group index $a$" is meaningless.
- $\mathcal L_{\text{EDL}}(\mathbf x_i,y_i)$ mixes scalar $y_i$ and one-hot $\mathbf y_i$. $\mathcal I[y,\mathbf p|\mathbf x]$ and $\mathcal I(\mathbf x)$ are used interchangeably. $w_a$ is called a "sensitivity weight" although it is a clipping weight.
- Algorithm 1 uses $N_{tr}$ without defining it and omits the train/calibration split.
- $\bar{|\mathcal C|}$ renders poorly. Use $\overline{|\mathcal C|}$ or $\mathbb E|\mathcal C|$.
- Module names are inconsistent: D-EDL vs EDL; REF-DP vs SC-DP; PC-PAT vs "SWD" in Table 2 (stale name); "Information-Theoretic DCDA" vs "DCDA".
- "Cov. Gap" in Table 1 becomes "Cov. Disparity" in Table 2.
- Equalized Odds is defined as the mean of the absolute TPR and FPR gaps. Hardt's EO requires both to vanish, so a max is more conventional.
- Predictive parity (Chouldechova, PPV equality) and calibration within groups (Kleinberg) are conflated.

**LaTeX and tables**
- Check that `amsthm` does not conflict with elsarticle's built-in theorem and proof handling.
- `\footnotesize{...}` after the tabular should be a `threeparttable`/`tablenotes` footnote.
- `\resizebox{\textwidth}` on a 10-column table at 12pt shrinks the text. Consider `\scriptsize` and a split into two tables.
- The set-size arrow ($\downarrow$) is misleading without joint coverage.
- Bolding vacuous 0.000* entries conflicts with the footnote calling them deceptive.
- Table 1 mixes 2- and 3-decimal precision in "mean ± std".
- The ablation caption says "best performance for each metric", but the full model is best in only one column.

**Prose**
- "premier international regulatory guidelines", "subgroup gradient preservation" and "harmonization of the Trilemma" are inflated.
- "Equitable" appears in the shortened title but "fair" elsewhere.
- Contribution 2 presents a textbook DP-SGD sensitivity argument as a proof.
- "Hyperbolic Riemannian curvature" should be removed or derived.
- The same citation (patel2022towards) is used both for explanation faithfulness and for privacy auditing, which is suspicious. tibshirani2019conformal is a covariate-shift paper cited as the generic CP reference. Charpentier 2020 is mis-described.
- Esipova et al. is ICLR 2023, not 2022.

**References (bib not inspected).** Cited keys cluster in 2024 and earlier. Given a date of October 2026, consider adding:
- 2025–2026 DP-conformal and DP-fairness work.
- Critical EDL analyses (Shen et al. 2024; Jürgens et al. 2024).
- Lowy et al. 2023; Jagielski et al. 2019; Cummings et al. 2019; Mozannar et al. 2020.
- FUTURE-AI (BMJ 2025, not the 2023 preprint).
- FDA's 2025 draft guidance on AI-enabled device lifecycle.
- Primary data citations: Detrano 1989 for Heart and Smith 1988 for Pima.

**Statements**
- Ethics: the DUA status of MIMIC/eICU-derived parameters is not addressed.
- Limitations omit all of the following: the theorem gaps, tiny samples, group attribute required at inference, ε magnitude, hyperparameter privacy cost, K=2 only, exchangeability under deployment shift, and no clinician evaluation.

## 7. Final verdict and actionable revision checklist

**Recommendation: major revision.** The ideas are worth pursuing, but the privacy and coverage claims are currently wrong or unsupported, and the results do not carry the abstract's narrative.

**Must fix before any resubmission**
1. Make PC-PAT genuinely per-sample (learnable, public or privately estimated prototypes) and re-prove Theorem 1, or drop the claim and Highlight 3.
2. Replace $n_a$ in $\tau_a$, $\lambda_a$ and the cap logic with a privatized quantity. Account for the marginal threshold. Correct the Bernstein constant ($2\sqrt{ML}$), handle the noisy normalizer and truncation, and state the end-to-end $(\epsilon,\delta)$ across training and calibration.
3. Remove the cap from the theorem or state honestly that the deployed procedure is a heuristic without PAC guarantee. Show $\tau_a$ for every cohort size actually used.
4. Replace the "epidemiological priors" with external, citable values or data-independent choices. Report $C_{\max}/C_0$ explicitly and add a matched-noise baseline (uniform $C_{\max}$, plus tuned DP-SGD).
5. Apply the identical conformal procedure to all baselines. Report marginal, per-group and per-class coverage, and singleton and empty-set rates.
6. Add strong baselines (logistic regression, GBM, DP-LR, DPSGD-F, Lowy, Tran) and compare on every metric. Re-tune baselines fairly.
7. Report CIs and paired tests using repeated CV (≥10×5). Report fairness and coverage with test-set sizes by group. Increase minority counts or aggregate over repeats.
8. Redo the ablation with retraining-noise controls and add cap/shrinkage, EDL and PAT-vs-SWD rows. Verify the 0.318 duplicate. Explain why every ablation improves EO.
9. Fix the text–table contradictions: DP-SGD ECE on Heart, the radar-chart "encloses" claim, the "all baseline DP models collapse" claim, the AUC/Brier ranges and the missing singleton rates.
10. Rewrite the abstract and conclusions around what the data supports. Heart gives an SP gain with worse EO, ECE and MIA. Pima gives an EO gain. EHR gives no gain over DP-EDL.

**Strongly recommended**
- Remove "Riemannian" or derive a Riemannian-norm clip. Derive $w_a$ from first principles.
- Replace the single-baseline DCDA with sanity checks and baseline comparisons, report Channel B infidelity and completeness error, and fix the categorical baselines (including `fasting blood sugar` and `chest pain` coding).
- Redefine the vacuity threshold or report abstention rates. Use one epistemic measure consistently.
- Document the EHR simulator fully (prevalence, SOFA handling, missingness, disparity injection) and run on real credentialed data or a real multi-site cohort.
- Discuss label leakage in Statlog `ca`/`thal`. Add an ε sweep, reliability diagrams and decision-curve analysis.
- Address the sensitive attribute needed at inference.
- Fix the highlights (≤85 characters), shorten the abstract, unify the notation, and fix the table footnote and bolding conventions.
- Update the references and the regulatory dates.

**Housekeeping you may want to act on (not part of the review).**
- The Generative-AI declaration names a Claude model. Elsevier requires AI-use disclosure, but your global rule says no Claude identity in the project. You need to decide which governs.
- I made no edits and no commits. If you want, I can draft a point-by-point revision plan or implement the fixes in `src/` and `manuscript/main.tex`, starting with the PC-PAT prototype handling and the conformal calibration.
