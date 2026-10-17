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

**"TRACER-Med: Evidential Deep Learning with Differentially Private Subgroup-Aware Optimization, Prototype Alignment, and Conformal Prediction for Clinical Decision Support"**  

for consideration as a regular research article in *Journal of Biomedical Informatics*.  

---

### Clinical Motivation and Scope

Clinical decision support models must protect patient privacy, perform comparably across demographic groups, report calibrated uncertainty, and be explainable. These requirements interact: gradient clipping and noise in differentially private training can change calibration and group performance, and private conformal calibration is difficult on small cohorts. The manuscript studies a framework, TRACER-Med, that combines an evidential Dirichlet network, differentially private optimization with subgroup-aware clipping, per-sample class-prototype alignment, private conformal calibration, and dual-channel attribution of disease risk and epistemic doubt, and it reports empirically which components help and which do not.

---

### Methodological Novelty

The contribution is a combination of known components with analysis, not a new learning paradigm. Existing methods address privacy and subgroup fairness together (DPSGD-F, PF-LD), or uncertainty without privacy (evidential learning, conformal prediction), or privacy for coverage alone (private conformal quantiles), and none of them also explains epistemic doubt separately from disease risk. TRACER-Med couples evidential learning with private optimization whose group clipping weights are derived from the Dirichlet Fisher metric and estimated by an accounted private release, per-sample prototype alignment that preserves the sensitivity analysis, and private conformal calibration with a coverage theorem. The manuscript states which of these statements were verified numerically and which were corrected during that audit, and compares the method with DPSGD-F implemented from its published algorithm and with an adaptation of PF-LD.

---

### Empirical Validation and Regulatory Mapping

We evaluate TRACER-Med on four cohorts: Statlog Heart Disease ($N=270$), Pima Indians Diabetes ($N=768$), a real 20,000-patient readmission cohort from the UCI Diabetes 130-US Hospitals data, and a simulated inpatient EHR cohort ($N=10{,}000$), against ERM, DP-SGD, DPSGD-F, an adaptation of the private fairness method PF-LD, evidential learning and DP-EDL, with an exact Poisson-subsampled RDP accountant, a common decision operating point, a noise-multiplier sweep and ablations. The results are mixed, and the manuscript reports them as such. TRACER-Med has markedly better calibration than the private baselines on the two larger cohorts at every noise level tested, and low empirical membership-inference AUC. It does not reduce Equalized Odds disparity relative to the private baselines, an adapted PF-LD baseline reduces disparity more, subgroup-calibrated clipping shows no consistent benefit, and the private conformal layer is vacuous or undercovers on small cohorts. The manuscript also contains a numerical audit of its analytical statements, which corrected several claims of an earlier version, and releases the code. Finally, we map the framework to the US FDA's Good Machine Learning Practice (GMLP), the FDA's December 2024 guidance on Predetermined Change Control Plans (PCCP), and Articles 9–15 of the European Union AI Act.

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
