# Submission Record & Milestone Log

**Date:** October 15, 2026  
**Status:** Rejected (Manuscript AIIM-D-26-06999, Oct 16, 2026)  
**Previous Journal:** *Artificial Intelligence in Medicine* (Elsevier), rejected
**Next Target Journal:** *Journal of Biomedical Informatics* (Elsevier)  
**Editor-in-Chief:** Prof. Zhengxing Huang  

---

### Manuscript Details

- **Title:** TRACER-Med: Trustworthy Deep Learning for Differentially Private, Equitable, and Uncertainty-Calibrated Clinical Decision Support
- **Author:** Dr. Quang-Vinh Dang
- **Affiliation:** School of Computing and Innovative Technologies, British University Vietnam (BUV)
- **Email:** [vinh.dq4@buv.edu.vn](mailto:vinh.dq4@buv.edu.vn)
- **Manuscript Length:** 35 pages (inclusive of figures, tables, and references)
- **References:** 60 fully verified authentic citations (with DOIs and PubMed IDs)
- **Keywords:** Trustworthy Machine Learning; Differential Privacy; Algorithmic Fairness; Evidential Deep Learning; Conformal Prediction; Clinical Decision Support

---

### Submitted Package Files

1. **Main Manuscript:**
   - PDF: [`manuscript/main.pdf`](manuscript/main.pdf)
   - Source: [`manuscript/main.tex`](manuscript/main.tex)
   - Bibliography: [`manuscript/references.bib`](manuscript/references.bib)
   - Figures: `manuscript/figures/*.pdf`
2. **Cover Letter:**
   - PDF: [`manuscript/cover_letter.pdf`](manuscript/cover_letter.pdf)
   - Source: [`manuscript/cover_letter.tex`](manuscript/cover_letter.tex)
   - Text/Markdown: [`manuscript/cover_letter.md`](manuscript/cover_letter.md)
3. **Research Highlights:**
   - PDF: [`manuscript/highlights.pdf`](manuscript/highlights.pdf)
   - Source: [`manuscript/highlights.tex`](manuscript/highlights.tex)
   - Text: [`manuscript/highlights.txt`](manuscript/highlights.txt)

---

### Highlight Points (Compliant with Elsevier $\le 85$ Characters Requirement)

1. `Formulates clinical Privacy-Fairness-Uncertainty Trilemma under DP.` (67 chars)
2. `Derives Riemannian Dirichlet-Fisher DP-SGD from information geometry.` (68 chars)
3. `Proves strict per-sample record privacy for class prototype transport.` (70 chars)
4. `Establishes finite-sample sub-exponential PAC conformal bounds.` (63 chars)
5. `Dual-channel Dirichlet XAI isolates disease risk from clinical doubt.` (69 chars)

---

### Suggested Reviewers Provided

1. **Prof. Georgios Kaissis** (Technical University of Munich / Imperial College London) — `g.kaissis@tum.de`  
   *Domain:* Biomedical Differential Privacy & Federated Learning
2. **Prof. Marzyeh Ghassemi** (Massachusetts Institute of Technology, MIT) — `mghassem@mit.edu`  
   *Domain:* Clinical Machine Learning, Algorithmic Fairness & Health Disparities
3. **Prof. Murat Sensoy** (Ozyegin University) — `murat.sensoy@ozyegin.edu.tr`  
   *Domain:* Evidential Deep Learning & Uncertainty Quantification
4. **Prof. Anastasios Angelopoulos** (University of California, Berkeley) — `angelopoulos@berkeley.edu`  
   *Domain:* Conformal Prediction & Private Risk-Controlling Prediction Sets

---

### Key Innovations Documented in Submission

- **REF-DP:** Closed-form inverse Fisher information $\mathcal{G}^{-1}(\boldsymbol{\alpha})$ via Sherman–Morrison formula on Dirichlet statistical manifold; subgroup-calibrated gradient clipping preserving protected class representations under Rényi Differential Privacy.
- **PC-PAT:** Class-conditional prototypical anchor transport with orthogonal angular margins; contracts $\mathcal{W}_2(\mathbb{P}_{0, y}, \mathbb{P}_{1, y}) \to 0$ without marginal demographic parity distortions, preserving strict per-sample DP.
- **DP-GCCP:** Group-conditional conformal calibration under Stratified Bounded Replacement Adjacency ($\Delta_1 = 2$); finite-sample sub-exponential PAC coverage and Empirical-Bayes shrinkage preventing prediction set collapse.
- **DCDA:** Axiomatic path-integrated dual-channel attribution relative to standardized physiological baselines; disentangles disease risk from epistemic clinical doubt.
- **Regulatory Alignment:** Formal mapping to FDA GMLP Guiding Principles, FDA Predetermined Change Control Plans (PCCP Dec 2024), and EU AI Act Articles 9–15.

---

### Tracking & Next Steps

| Stage | Expected Status | Projected Timeline | Notes |
| :--- | :--- | :--- | :--- |
| **Stage 1** | Submitted to Journal / With Editor | Days 1–7 | Initial technical check & Editor assignment |
| **Stage 2** | Under Review | Weeks 2–8 | Reviewer invitation and report collection |
| **Stage 3** | First Editorial Decision | Weeks 8–12 | Major/Minor revision or decision notice |
| **Stage 4** | Revision & Rebuttal | Upon decision | Address reviewer queries systematically |

---

### Editorial Decision (Oct 16, 2026)

*Artificial Intelligence in Medicine* rejected the manuscript without review. The Editor-in-Chief's comment: the journal requires novelty in methodological and/or theoretical content in AI and computer science, shown through experimental evaluation, comparison with other proposals and explicit discussion of novelty. Applying well-known or published techniques to medical data is not regarded as original research for the journal.

Next step: reposition the contribution around its methodological novelty (REF-DP, PC-PAT, DP-GCCP, DCDA) and resubmit to a suitable venue.


---

# Submission 2: BMC Medical Informatics and Decision Making

**Date submitted:** October 17, 2026
**Status:** Submitted (manuscript number: to be added when received)
**Article type:** Research article
**Title:** TRACER-Med: differentially private evidential learning with prototype alignment and conformal prediction for clinical risk prediction

**Authors:** Quang-Vinh Dang (submitting author, British University Vietnam), Dat Le (corresponding author, Ho Chi Minh City University of Economics and Finance), Minh Ngoc Dinh (Millennia Education)

**Submitted files** (in `manuscript/bmc/`): `main_bmc.tex` with `main_bmc.bbl` and `main_bmc.pdf`; figures in `manuscript/figures/`; `additional_file_1.csv` (main results); `additional_file_2.csv` (noise sweep); `cover_letter_bmc.pdf`.
**Author contributions statement** (entered in the submission form): Q.-V.D. conceived and designed the study, implemented the methods, ran the experiments and analyses, prepared the figures and tables, and wrote the main manuscript text. D.L. contributed to the study design and methodology, interpreted the results, and critically revised the manuscript. M.N.D. contributed to data preparation and validation of the analyses, and critically revised the manuscript. All authors reviewed and approved the final manuscript.
**Declarations:** no funding, no competing interests, no ethics approval required (public de-identified and synthetic data).
**Archived code version cited in the manuscript:** git commit `e8ed36b`.

**Previous submission:** *Artificial Intelligence in Medicine* (Elsevier), rejected without review on October 16, 2026 (see above).

### Open items
- Add a Zenodo (or similar) DOI for the code and replace the commit reference in the Availability section if the journal asks for it.
- Reference list uses full journal names; BMC asks for Index Medicus/MEDLINE abbreviations. Fix if requested at revision.
- No reviewers were suggested in the cover letter.
- Keep `manuscript/bmc/main_bmc.tex` (generated from `manuscript/main.tex`) in sync when preparing a revision.
- Key limitations stated in the paper: weak privacy budgets at noise multiplier 0.8; no Equalized Odds gain over the private baselines; REF-DP shows no consistent benefit; private conformal sets are uninformative on small cohorts; one real cohort with weak signal; PF-LD is an adaptation tuned on the evaluation splits.

### BMC technical check (October 20, 2026)
Manuscript ID f9b1685e-63ac-430d-9b2c-97bc0c34abde v1.0 was returned by editorial support: an editable manuscript file is required. Fix: upload `manuscript/bmc/bmc_latex_source.zip` (main_bmc.tex, .bbl, class and style files, references.bib, figures/) as the main manuscript file. Content unchanged; only the figure and bibliography paths were made relative to the zip root. Compiles standalone with pdflatex + bibtex.
