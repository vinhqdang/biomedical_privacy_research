# Cover Letter

**Dr. Quang-Vinh Dang**  
School of Computing and Innovative Technologies  
British University Vietnam  
Ecopark Township, Hung Yen, Vietnam  
Email: [vinh.dq4@buv.edu.vn](mailto:vinh.dq4@buv.edu.vn)  

October 16, 2026  

**The Editor-in-Chief**  
*Journal of Biomedical Informatics*  
Elsevier  

**Subject: Submission of Original Research Manuscript to *Journal of Biomedical Informatics***  

Dear Editor-in-Chief and Members of the Editorial Board,  

I am pleased to submit our original research manuscript titled:  

**"TRACER-Med: Trustworthy Deep Learning for Differentially Private, Equitable, and Uncertainty-Calibrated Clinical Decision Support"**  

for consideration as a regular research article in *Journal of Biomedical Informatics*.  

---

### Clinical Motivation and Theoretical Contributions

The deployment of deep neural networks in real-world clinical decision support systems (CDSS) is governed by strict biomedical, ethical, and regulatory demands: patient privacy under HIPAA and GDPR, demographic fairness across protected patient sub-populations, calibrated uncertainty quantification to prevent overconfident diagnostic errors, and actionable clinical explainability.

In contemporary clinical informatics, hospital data science teams frequently attempt a modular, fragmented assembly—combining standard Differentially Private SGD (DP-SGD), post-hoc fairness thresholding, evidential learning, and single-channel saliency maps. In this paper, we demonstrate that this modular patchwork fundamentally breaks down, formalizing the **Privacy–Fairness–Uncertainty Trilemma**:
1. Uniform Euclidean gradient clipping in DP-SGD disproportionately truncates high-variance minority gradients, acting as an equity penalty that widens demographic diagnostic disparities.
2. Privacy perturbation suppresses Dirichlet evidence accumulation, corrupting probability calibration and inflating Expected Calibration Error.
3. Post-hoc fairness adjustments distort probability calibration, while private conformal calibration over small clinical cohorts risks degenerating into vacuous prediction sets.

To resolve these interconnected challenges, we present **TRACER-Med**, an organically unified framework that bridges differential privacy, algorithmic fairness, evidential uncertainty quantification, and path-integrated explainability through four co-designed pillars:
- **Riemannian Evidential-Fisher Subgroup-Calibrated DP-SGD (REF-DP):** Exploiting information geometry on the Dirichlet statistical manifold, we derive the exact closed-form inverse Fisher Information Metric $\mathcal{G}^{-1}(\boldsymbol{\alpha})$ via the Sherman–Morrison formula, formulating Riemannian Dirichlet natural gradients and adaptive cohort clipping thresholds that prevent minority gradient suppression under Rényi Differential Privacy.
- **Privatized Class-Conditional Prototypical Anchor Transport (PC-PAT):** Rather than enforcing marginal demographic parity which distorts disease base rates, we formulate class-conditional alignment toward learnable diagnostic prototypes with an orthogonal angular margin constraint, proving that this contracts the demographic 2-Wasserstein transport distance $\mathcal{W}_2(\mathbb{P}_{0, y}, \mathbb{P}_{1, y}) \to 0$ while preserving strict per-sample record independence.
- **Differentially Private Group-Conditional Conformal Prediction (DP-GCCP):** Under Stratified Bounded Replacement Adjacency ($\Delta_1 = 2$), we prove a finite-sample sub-exponential PAC coverage theorem and establish an Empirical-Bayes shrinkage estimator that generates compact, informative prediction sets without vacuity collapse.
- **Axiomatic Dual-Channel Dirichlet Attribution (DCDA):** Satisfying Completeness, Linearity, and Implementation Invariance, DCDA disentangles features driving posterior disease risk from features driving epistemic clinical doubt relative to standardized physiological reference baselines.

---

### Methodological Novelty

The contribution is methodological. Existing methods address privacy and subgroup fairness together (DPSGD-F, fair private empirical risk minimization), or uncertainty without privacy (evidential learning, group-conditional conformal prediction), or privacy for coverage alone (private conformal quantiles). None addresses all three requirements, nor explains epistemic doubt separately from disease risk. TRACER-Med derives subgroup clipping thresholds from the Fisher metric of the Dirichlet output manifold, aligns class-conditional prototypes without cross-sample gradient coupling, and calibrates conformal sets per cohort with a finite-sample PAC guarantee under bounded replacement adjacency. Each component carries a formal guarantee, stated and proved in the manuscript, and the combined system is compared against ERM, DP-SGD, Fair-ERM, evidential learning and DP-EDL baselines with ablations. The manuscript discusses novelty relative to prior work in a dedicated subsection of the Related Work section.

---

### Empirical Validation and Regulatory Mapping

We validate TRACER-Med across three clinical benchmarks: Statlog Heart Disease ($N=270, D=12$), Pima Indians Diabetes ($N=768, D=8$), and a simulated acute inpatient EHR cohort ($N=10{,}000, D=15$). Across all benchmarks, TRACER-Med achieves balanced multi-objective performance, mitigating demographic error disparities while matching underlying epidemiological prevalence gaps, avoiding conformal set collapse, and demonstrating empirical resilience against membership inference attacks (MIA AUC $\approx 0.505$). Finally, we map our framework directly to international regulatory frameworks, including the US FDA's Good Machine Learning Practice (GMLP), the FDA's December 2024 guidance on Predetermined Change Control Plans (PCCP), and Articles 9–15 of the European Union AI Act.

---

### Declarations and Compliance

In accordance with Elsevier's editorial policies:
- This manuscript represents original work that has not been published previously and is not under consideration for publication elsewhere.
- Publication has been approved by the author and the host institution.
- The author has no competing financial interests or personal relationships that could influence this work.
- Secondary analysis of de-identified open-source data did not involve human participants; IRB review was not required under 45 CFR 46.102.
- All data sources, model parameters, and mathematical proofs are fully documented in the manuscript.

---

### Suggested Expert Reviewers

To assist the editorial office in selecting qualified peer reviewers, we respectfully suggest the following independent experts with expertise in biomedical privacy, algorithmic fairness, evidential deep learning, and clinical decision support:

1. **Prof. Georgios Kaissis** (Technical University of Munich / Imperial College London)  
   Email: `g.kaissis@tum.de`  
   *Expertise:* Privacy-preserving medical machine learning, federated learning, biomedical differential privacy.

2. **Prof. Marzyeh Ghassemi** (Massachusetts Institute of Technology, MIT)  
   Email: `mghassem@mit.edu`  
   *Expertise:* Clinical machine learning, algorithmic fairness, health disparities in electronic health records.

3. **Prof. Murat Sensoy** (Ozyegin University)  
   Email: `murat.sensoy@ozyegin.edu.tr`  
   *Expertise:* Evidential deep learning, subjective logic, uncertainty quantification.

4. **Prof. Anastasios Angelopoulos** (University of California, Berkeley)  
   Email: `angelopoulos@berkeley.edu`  
   *Expertise:* Distribution-free conformal prediction, private quantile estimation, risk-controlling prediction sets.

---

Thank you for your time, consideration, and editorial stewardship of this manuscript. I look forward to hearing from you.

Sincerely,  

**Dr. Quang-Vinh Dang**  
British University Vietnam  
Corresponding Author  
Email: [vinh.dq4@buv.edu.vn](mailto:vinh.dq4@buv.edu.vn)  
