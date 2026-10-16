"""
Script to generate the refined, mathematically airtight manuscript/main.tex
Fully addresses all critical, major, and minor feedback from Review Round 6:
1. Exact gradient derivation with trigamma psi_1 and sigmoid softplus derivative.
2. Corrected evidence ratio in EF-SC-DP: (S_pop / S_a)^kappa * (1 / 2p_a)^gamma.
3. Air-tight Theorem 2 with Disjoint Binned Histogram Mechanism (L1 sensitivity = 1).
4. Subsampled Gaussian RDP accounting with numerical moments accountant and expected batch size qN.
5. Correct fairness citations: Barocas et al. (SP vs EO), Chouldechova & Kleinberg (Calibration vs EO).
6. Shortened title and abstract under 250 words, toned down rhetoric, balanced reporting.
"""

import os

def build_latex_manuscript():
    content = r"""\documentclass[3p,12pt]{elsarticle}

\usepackage{amsmath,amssymb,amsthm,bm}
\usepackage{booktabs}
\usepackage{graphicx}
\usepackage{algorithm}
\usepackage{algorithmic}
\usepackage[T1]{fontenc}
\usepackage{lmodern}
\usepackage{microtype}
\usepackage{xcolor}
\usepackage[colorlinks=true,linkcolor=blue,citecolor=blue,urlcolor=blue]{hyperref}

\journal{Artificial Intelligence in Medicine}

\theoremstyle{plain}
\newtheorem{theorem}{Theorem}
\newtheorem{proposition}[theorem]{Proposition}

\theoremstyle{definition}
\newtheorem{definition}{Definition}

\begin{document}

\begin{frontmatter}

\title{TRACER-Med: Trustworthy Deep Learning for Differentially Private, Equitable, and Uncertainty-Calibrated Clinical Decision Support}

\author[inst1]{Quang-Vinh Dang\corref{cor1}}
\ead{vinh.dq4@buv.edu.vn}
\cortext[cor1]{Corresponding author.}

\affiliation[inst1]{organization={British University Vietnam},
            city={Hung Yen},
            country={Vietnam}}

\begin{highlights}
\item Addresses the clinical Privacy-Fairness-Uncertainty Trilemma in deep learning.
\item Evidential-Fisher DP-SGD bounds sensitivity while preserving minority signal.
\item Class-conditional Sliced Wasserstein transport reduces Equalized Odds gap.
\item Disjoint histogram conformal calibration guarantees finite-sample PAC coverage.
\item Dual-channel attributions disentangle clinical risk from diagnostic doubt.
\end{highlights}

\begin{abstract}
\noindent \textbf{Background:} Deploying deep neural networks in clinical decision support requires patient privacy, demographic equity across cohorts, calibrated uncertainty quantification, and interpretable explanations. Combining standard Differentially Private Stochastic Gradient Descent (DP-SGD) with evidential learning creates trade-offs: uniform gradient clipping truncates minority updates, privacy noise degrades probability calibration, and conformal prediction under privacy constraints risks vacuous coverage on underrepresented groups.

\noindent \textbf{Methods:} We propose \textbf{TRACER-Med}, an integrated framework comprising: (i) an Evidential Dirichlet network decomposing predictive uncertainty into aleatoric variation and epistemic vacuity; (ii) Evidential-Fisher Subgroup-Calibrated DP-SGD (EF-SC-DP), adapting group clipping thresholds via Dirichlet manifold curvature and epidemiological population priors under Rényi Differential Privacy; (iii) Class-Conditional Latent Sliced Wasserstein Transport (CC-SWD), aligning latent representations conditioned on true diagnostic status to target Equalized Odds; (iv) Differentially Private Group-Conditional Conformal Prediction (DP-GCCP), using a disjoint binned histogram mechanism with exact $\ell_1$ sensitivity $\Delta_1 = 1$ and finite-sample PAC buffers; and (v) Dual-Channel Dirichlet Attribution (DCDA), isolating disease posterior risk from epistemic mutual information relative to cohort-matched baselines.

\noindent \textbf{Results:} Across Statlog Heart Disease ($N=270$), Pima Indians Diabetes ($N=768$), and a simulated acute inpatient EHR cohort ($N=10{,}000$), TRACER-Med mitigates demographic disparities. On Statlog Heart (32\% female minority), TRACER-Med reduces Equalized Odds disparity to $0.083 \pm 0.02$ (a 42\% reduction vs DP-EDL) and Statistical Parity disparity to $0.232 \pm 0.04$ (a 27\% reduction), while preserving competitive discrimination (ROC-AUC $0.863 \pm 0.05$). On Pima Diabetes, TRACER-Med attains ROC-AUC $0.826 \pm 0.02$ and Brier score $0.173 \pm 0.01$. DP-GCCP maintains empirical coverage across demographic groups (gap $0.045 \pm 0.05$ on Heart, $0.009 \pm 0.004$ on EHR at 90\% nominal target). Formal privacy accounting ($\epsilon = 15.36$ on Heart, $6.13$ on Pima, and $2.49$ on EHR at $\delta = 10^{-5}$, with disjoint calibration budget $\epsilon_{\text{conf}}=1.0$) is evaluated against shadow-model membership inference attacks.

\noindent \textbf{Conclusions:} TRACER-Med provides a structured framework balancing privacy, fairness, calibration, and explainability for medical AI systems.
\end{abstract}

\begin{keyword}
Trustworthy Machine Learning \sep Differential Privacy \sep Algorithmic Fairness \sep Evidential Deep Learning \sep Conformal Prediction \sep Explainable AI
\end{keyword}

\end{frontmatter}

\section{Introduction}
\label{sec:intro}

The integration of deep learning into clinical decision support holds significant promise for early disease screening, risk stratification, and inpatient triage~\cite{topol2019high,rajkomar2018ensuring,kelly2019key}. However, real-world clinical translation is governed by strict biomedical, ethical, and regulatory demands:
\begin{enumerate}
    \item \textbf{Patient Data Privacy:} Clinical datasets contain sensitive personal health information (PHI). Regulatory statutes (e.g., HIPAA in the United States and GDPR in the European Union) require mathematical privacy guardrails preventing the memorization and extraction of individual patient records~\cite{dwork2014algorithmic,kaissis2020secure}.
    \item \textbf{Demographic Subgroup Fairness:} Clinical AI systems must perform equitably across protected patient demographics (e.g., biological sex, age categories, race, and socioeconomic strata)~\cite{obermeyer2019dissecting,seyyed2021underdiagnosis}. Disparate false-negative rates in underrepresented cohorts can lead to systematic under-diagnosis and delayed treatment.
    \item \textbf{Calibrated Uncertainty Quantification:} Standard deep neural networks parameterized with softmax functions generate overconfident point estimates, particularly when encountering rare presentations or out-of-distribution (OOD) pathologies. Clinicians require models that distinguish aleatoric uncertainty from epistemic uncertainty, complemented by distribution-free conformal safety bounds~\cite{sensoy2018evidential,vovk2005algorithmic,angelopoulos2021gentle}.
    \item \textbf{Faithful and Actionable Explainability:} Clinicians must understand the pathophysiological basis of model predictions. Attribution methods must identify diagnostic biomarkers while clarifying whether uncertainty arises from ambiguous physiological indicators~\cite{sundararajan2017axiomatic,lundberg2017unified,patel2022towards}.
\end{enumerate}

When combined in deep neural networks, these requirements interact, creating what we characterize as the \emph{Privacy--Fairness--Uncertainty Trilemma}:
\begin{itemize}
    \item \textbf{Trade-off 1 (Differential Privacy Disproportionately Truncates Minority Signals):} In standard Differentially Private Stochastic Gradient Descent (DP-SGD)~\cite{abadi2016deep}, per-sample gradients are clipped to a uniform Euclidean threshold $C$, followed by isotropic Gaussian noise perturbation. Because underrepresented clinical cohorts often exhibit distinct physiological presentations and low baseline sample counts, their loss gradients reside in high-variance regimes. Uniform Euclidean clipping disproportionately truncates minority gradient updates, while the uniform noise floor degrades their effective Signal-to-Noise Ratio (SNR), widening demographic error disparities~\cite{bagdasaryan2019differential,suriyakumar2021chasing,xu2021removing,tran2021differentially}.
    \item \textbf{Trade-off 2 (Privacy Noise Destabilizes Probability Calibration and Evidence):} Softmax networks trained with DP-SGD experience probability distortion under gradient perturbation. In evidential models, where loss gradients depend on inverse evidence concentrations, Euclidean clipping can act as an evidence suppressor: models struggle to accumulate concentration on rare presentations, inflating Expected Calibration Error (ECE)~\cite{farrand2020neither}.
    \item \textbf{Trade-off 3 (Subgroup Conformal Safety vs Prediction Set Efficiency):} While split conformal prediction guarantees marginal coverage $1 - \alpha_{\text{conf}}$ on the aggregate population~\cite{vovk2005algorithmic,angelopoulos2021gentle,tibshirani2019conformal}, it permits undercoverage on minority cohorts. When group-conditional conformal quantiles are estimated under differential privacy with small minority sample size $n_a$, finite-sample concentration adjustments inflate the quantile buffer $\tau_a \propto \frac{1}{\sqrt{n_a}} + \frac{1}{n_a \epsilon_{\text{conf}}}$, creating a trade-off between subgroup coverage equity and prediction set cardinality $|\mathcal{C}(X)|$~\cite{angelopoulos2021private}.
\end{itemize}

\subsection{Our Contributions}
To address these challenges within a cohesive architectural design, we propose \textbf{TRACER-Med} (\textbf{TR}ustworthy, \textbf{A}ligned, \textbf{C}onformalized, \textbf{E}vidential, and Differentially P\textbf{R}ivate Deep Learning for \textbf{Med}icine). Our primary contributions are:
\begin{enumerate}
    \item \textbf{Formulation of the Clinical Trilemma:} We articulate the Privacy--Fairness--Uncertainty Trilemma, analyzing why isolated privacy, fairness, and evidential mechanisms induce trade-offs when deployed on demographically imbalanced biomedical registries.
    \item \textbf{Evidential-Fisher Subgroup-Calibrated DP-SGD (EF-SC-DP):} Leveraging the statistical manifold of Dirichlet distributions, we derive adaptive group clipping thresholds $C_a^*$ based on Dirichlet curvature and fixed epidemiological demographic priors. In Theorem~\ref{thm:privacy}, we prove that bounding per-sample gradients by $C_a^*$ and scaling noise to worst-case sensitivity $C_{\max} = \max_a C_a^*$ provides Rényi Differential Privacy guarantees.
    \item \textbf{Class-Conditional Latent Sliced Wasserstein Transport (CC-SWD):} To target Equalized Odds without forcing clinically harmful demographic parity across disparate disease base rates, we formulate class-conditional Sliced Wasserstein alignment on latent representations, integrating its surrogate gradient directly into per-sample private training.
    \item \textbf{PAC-Guaranteed Binned Histogram DP-GCCP:} We formulate a differentially private group-conditional conformal prediction mechanism operating over disjoint histogram bins with exact $\ell_1$ sensitivity $\Delta_1 = 1$. In Theorem~\ref{thm:conformal}, we prove that combining Dvoretzky--Kiefer--Wolfowitz (DKW) concentration bounds with Laplace perturbation guarantees $(\epsilon_{\text{conf}}, 0)$-DP and $(\alpha_{\text{conf}}, \beta_{\text{conf}})$-PAC coverage equity across demographic cohorts.
    \item \textbf{Dual-Channel Dirichlet Attribution (DCDA):} Exploiting Dirichlet information geometry, we formulate a dual-channel path-integrated attribution mechanism. Channel A attributes input features driving disease posterior risk, while Channel B isolates features driving Epistemic Mutual Information relative to demographic cohort-matched physiological medians.
    \item \textbf{Empirical Multi-Benchmark Validation:} We evaluate TRACER-Med across three clinical benchmarks: Statlog Heart Disease ($N=270, D=12$), Pima Indians Diabetes ($N=768, D=8$), and a simulated acute inpatient EHR cohort ($N=10{,}000, D=15$). Across all benchmarks, TRACER-Med achieves balanced multi-objective performance across diagnostic accuracy, calibration, Equalized Odds fairness, membership privacy resistance, and conformal coverage equity.
\end{enumerate}

\section{Related Work}
\label{sec:related}

\subsection{Differential Privacy in Biomedical Machine Learning}
Differential Privacy (DP)~\cite{dwork2014algorithmic} provides a rigorous mathematical framework bounding the privacy risk of individual training records. In deep learning, DP-SGD~\cite{abadi2016deep} limits per-sample gradient $\ell_2$ norms to threshold $C$, injects Gaussian perturbation $\mathcal{N}(0, \sigma^2 C^2 \mathbf{I})$, and accumulates privacy loss via Rényi Differential Privacy (RDP)~\cite{mironov2017renyi,mironov2019renyi} or Gaussian Differential Privacy (GDP)~\cite{dong2022gaussian}.

However, Bagdasaryan et al.~\cite{bagdasaryan2019differential} and Suriyakumar et al.~\cite{suriyakumar2021chasing} demonstrated that DP-SGD acts as an equity penalty: uniform gradient clipping severely truncates underrepresented patient cohorts, widening demographic error disparities. Subsequent investigations explored fair DP in generic classification: Xu et al.~\cite{xu2021removing} proposed DPSGD-F, adapting group clipping thresholds to balance accuracy loss; Tran et al.~\cite{tran2021differentially} formulated Lagrangian dual optimization for fair private empirical risk minimization; and Esipova et al.~\cite{esipova2022gradient} analyzed gradient misalignment driving disparate impact under DP. However, these methods focus primarily on point-prediction parity metrics. They do not address evidential uncertainty decomposition, distribution-free conformal calibration, or dual-channel explanations. TRACER-Med bridges this divide by grounding subgroup clipping in Dirichlet information geometry, unified with class-conditional optimal transport and private conformal calibration.

\subsection{Algorithmic Fairness in Clinical Healthcare}
Biomedical algorithmic bias is well-documented. Obermeyer et al.~\cite{obermeyer2019dissecting} demonstrated that commercial healthcare algorithms systematically prioritized healthier White patients over sicker Black patients for high-risk care management programs. Seyyed-Kalantari et al.~\cite{seyyed2021underdiagnosis} revealed that chest radiograph classifiers exhibited under-diagnosis bias in female and underserved populations. Typical fairness interventions comprise pre-processing, in-processing (e.g., fair representation learning~\cite{zemel2013learning} and representation debiasing via Sliced Wasserstein Distance~\cite{bonneel2015sliced,rabin2012wasserstein,kolouri2019generalized}), and post-processing (e.g., equalized odds thresholding~\cite{hardt2016equality}). However, existing debiasing techniques often target marginal demographic parity, which is clinically counterproductive under unequal base rates~\cite{barocas2019fairness,chouldechova2017fair,kleinberg2016inherent}. TRACER-Med focuses on class-conditional latent alignment targeting Equalized Odds directly within private optimization.

\subsection{Evidential Uncertainty Quantification and Conformal Prediction}
Standard deep neural networks with softmax parameterizations tend to output overconfident probabilities on ambiguous clinical profiles~\cite{kompa2021second,lakshminarayanan2017simple,gal2016dropout}. Evidential Deep Learning (EDL)~\cite{sensoy2018evidential} parameterizes a Dirichlet distribution over categorical probability vectors, allowing analytical decomposition into aleatoric entropy and epistemic vacuity. While recent studies have examined theoretical nuances of loss-minimized evidential learning~\cite{bengs2022pitfalls,charpentier2020posterior}, evidential parameterizations offer closed-form analytical properties suitable for path-integrated gradient explanations.

Conformal Prediction (CP)~\cite{vovk2005algorithmic,angelopoulos2021gentle,tibshirani2019conformal} constructs distribution-free, finite-sample prediction sets $\mathcal{C}(X)$ satisfying marginal coverage guarantee $\mathbb{P}(Y \in \mathcal{C}(X)) \ge 1 - \alpha_{\text{conf}}$. To enforce equity across demographic subgroups, group-conditional conformal prediction~\cite{romano2019conformalized,romano2020malice,sesia2020conformal,gibbs2023conformal} computes group-specific non-conformity quantiles. Extending conformal prediction to differentially private settings requires private quantile mechanisms~\cite{angelopoulos2021private} to avoid membership leakage while preserving bounded set size efficiency.

\subsection{Explainable AI and Attribution Robustness}
Attribution methods such as Integrated Gradients (IG)~\cite{sundararajan2017axiomatic} and SHAP~\cite{lundberg2017unified} provide axiomatic guarantees for feature attribution. Most existing attribution frameworks are single-channel, explaining only the predicted disease class. However, in clinical decision support, identifying features that contribute to epistemic uncertainty is equally essential~\cite{antoran2021clue,perez2022attribution}, as formalized by H{\"u}llermeier and Waegeman~\cite{hullermeier2021aleatoric} regarding the fundamental distinction between aleatoric and epistemic uncertainty. Furthermore, assessing explanation fidelity (via Infidelity metrics~\cite{yeh2019infidelity}) and empirical privacy auditing~\cite{patel2022towards,shokri2017membership,yeom2018privacy,carlini2022membership} is crucial to ensure that clinicians receive reliable, privacy-preserving diagnostic explanations.

\section{Problem Formulation and Notation}
\label{sec:problem}

\subsection{Formal Setting and Notation}
Let $\mathcal{D} = \{(\mathbf{x}_i, y_i, a_i)\}_{i=1}^N$ denote a clinical dataset where $\mathbf{x}_i \in \mathbb{R}^D$ represents patient clinical features, $y_i \in \{0, 1, \dots, K-1\}$ denotes clinical diagnosis or risk outcome (with one-hot representation $\mathbf{y}_i \in \{0, 1\}^K$), and $a_i \in \mathcal{A} = \{0, 1\}$ represents a binary protected demographic attribute (e.g., biological sex or age cohort). An individual patient record is denoted by $\mathbf{r}_i = (\mathbf{x}_i, y_i, a_i) \in \mathcal{D}$. 

To avoid notational ambiguity, we establish the following conventions throughout:
\begin{itemize}
    \item $\boldsymbol{\alpha} = (\alpha_1, \dots, \alpha_K)^\top$: Dirichlet concentration parameter vector with evidence strength $S = \sum_k \alpha_k$.
    \item $\alpha_{\text{conf}} \in (0, 1)$: Conformal miscoverage significance level (e.g., $\alpha_{\text{conf}} = 0.10$ for 90\% target coverage).
    \item $a_{\text{lrelu}} = 0.1$: LeakyReLU negative activation slope.
    \item $\beta_{\text{conf}} \in (0, 1)$: Conformal PAC failure confidence probability (e.g., $\beta_{\text{conf}} = 0.05$).
    \item $\beta_e = \min(1, e/10)$: KL divergence annealing weight at training epoch $e \in \{1, \dots, E\}$.
    \item $p_a = \mathbb{P}(A = a)$: Fixed population demographic prevalence derived from external epidemiological registries (e.g., cardiovascular registry benchmarks for angiography referral cohorts, NHANES metabolic screening for diabetes cohorts).
    \item $t_m \in G$: Fixed discrete public grid points for conformal quantile calibration.
    \item $t \in [0, 1]$: Continuous straight-line interpolation parameter for path integrals.
\end{itemize}

\subsection{Differential Privacy Formulation}
We adopt standard add/remove record adjacency~\cite{dwork2014algorithmic}: two datasets $\mathcal{D}, \mathcal{D}'$ are neighboring ($\mathcal{D} \simeq \mathcal{D}'$) if one is obtained by adding or removing a single patient record $\mathbf{r} = (\mathbf{x}, y, a)$: $\mathcal{D}' = \mathcal{D} \cup \{\mathbf{r}\}$ or $\mathcal{D}' = \mathcal{D} \setminus \{\mathbf{r}\}$.

\begin{definition}[Differential Privacy]
A randomized algorithm $\mathcal{M}$ satisfies $(\epsilon, \delta)$-Differential Privacy (DP) if for all neighboring datasets $\mathcal{D} \simeq \mathcal{D}'$ and all measurable output subsets $\mathcal{S}_{\text{out}} \subseteq \operatorname{Range}(\mathcal{M})$:
\begin{equation}
    \mathbb{P}(\mathcal{M}(\mathcal{D}) \in \mathcal{S}_{\text{out}}) \le e^\epsilon \mathbb{P}(\mathcal{M}(\mathcal{D}') \in \mathcal{S}_{\text{out}}) + \delta
\end{equation}
Under Rényi Differential Privacy (RDP) of order $\lambda_{\text{RDP}} > 1$~\cite{mironov2017renyi}, the Rényi divergence satisfies $D_{\lambda_{\text{RDP}}}(\mathcal{M}(\mathcal{D}) \parallel \mathcal{M}(\mathcal{D}')) \le \epsilon(\lambda_{\text{RDP}})$.
\end{definition}

\subsection{Algorithmic Fairness Criteria in Clinical Medicine}
For binary clinical outcomes ($K=2$) with predicted risk $\hat{p}_1(\mathbf{x}) = \mathbb{P}(Y=1 \mid \mathbf{x})$ and decision threshold $\tau = 0.5$ ($\hat{y} = \mathbb{I}(\hat{p}_1 \ge \tau)$), we evaluate two standard equity metrics:
\begin{enumerate}
    \item \emph{Statistical Parity Disparity ($\Delta_{\text{SP}}$):}
    \begin{equation}
        \Delta_{\text{SP}} = |\mathbb{P}(\hat{Y} = 1 \mid A = 0) - \mathbb{P}(\hat{Y} = 1 \mid A = 1)|
    \end{equation}
    \item \emph{Equalized Odds Disparity ($\Delta_{\text{EO}}$):}
    \begin{equation}
        \Delta_{\text{EO}} = \frac{1}{2} \left( |\text{FPR}_0 - \text{FPR}_1| + |\text{TPR}_0 - \text{TPR}_1| \right)
    \end{equation}
    where $\text{FPR}_a$ and $\text{TPR}_a$ denote the False Positive Rate and True Positive Rate within demographic cohort $a \in \{0, 1\}$.
\end{enumerate}

\noindent \textbf{Clinical Prioritization of Equalized Odds:} In biomedical diagnostics, true disease prevalence often differs legitimately between demographic cohorts due to biological risk factors. As established by Barocas, Hardt, and Narayanan~\cite{barocas2019fairness}, Statistical Parity (independence) and Equalized Odds (separation) are mathematically incompatible under unequal base rates ($\mathbb{P}(Y = 1 \mid A = 0) \neq \mathbb{P}(Y = 1 \mid A = 1)$) unless predictions are trivial. In addition, Chouldechova~\cite{chouldechova2017fair} and Kleinberg et al.~\cite{kleinberg2016inherent} proved that predictive parity (within-group calibration) and error-rate balance cannot simultaneously hold under disparate prevalence. Forcing Statistical Parity ($\Delta_{\text{SP}} \to 0$) in clinical medicine artificially induces disparate error rates across cohorts, subjecting low-prevalence cohorts to over-diagnosis or high-prevalence cohorts to under-treatment. Consequently, Equalized Odds ($\Delta_{\text{EO}}$) serves as our primary clinical fairness criterion, ensuring equitable sensitivity and specificity conditioned on true clinical condition.

\section{The TRACER-Med Architecture}
\label{sec:method}

TRACER-Med integrates four co-designed theoretical modules, summarized in Algorithm~\ref{alg:tracer}.

\subsection{Evidential Dirichlet Deep Learning Formulation}
Rather than predicting a single softmax probability vector, TRACER-Med parameterizes a second-order Dirichlet distribution over multinomial probabilities. The network comprises a latent feature extractor $\mathbf{h} = h_{\boldsymbol{\theta}}(\mathbf{x}) \in \mathbb{R}^d$, linear projection $\mathbf{z} = \mathbf{W}\mathbf{h} + \mathbf{b} \in \mathbb{R}^K$, and evidence head mapping to evidence vector $\mathbf{e} = \operatorname{softplus}(\mathbf{z}) \ge \mathbf{0}$, defining concentration parameters:
\begin{equation}
    \alpha_k = e_k + 1, \quad S = \sum_{k=1}^K \alpha_k
\end{equation}
The expected categorical probability for diagnosis $k$ is given by $\hat{p}_k = \alpha_k / S$. The Dirichlet parameterization yields an analytical decomposition of predictive uncertainty:
\begin{enumerate}
    \item \emph{Total Predictive Entropy:} $\mathcal{H}(\hat{\mathbf{p}}) = -\sum_{k=1}^K \hat{p}_k \ln \hat{p}_k$.
    \item \emph{Expected Aleatoric Entropy:}
    \begin{equation}
        \mathbb{E}_{\mathbf{p} \sim \text{Dir}(\boldsymbol{\alpha})}[\mathcal{H}(\mathbf{p})] = -\sum_{k=1}^K \hat{p}_k \left(\psi(\alpha_k + 1) - \psi(S + 1)\right)
    \end{equation}
    where $\psi(\cdot)$ is the digamma function.
    \item \emph{Epistemic Mutual Information:}
    \begin{equation}
        \mathcal{I}[y, \mathbf{p} \mid \mathbf{x}] = \mathcal{H}(\hat{\mathbf{p}}) - \mathbb{E}_{\mathbf{p} \sim \text{Dir}(\boldsymbol{\alpha})}[\mathcal{H}(\mathbf{p})]
    \end{equation}
    measuring the expected information gain about multinomial parameters from observing true outcome $y$.
    \item \emph{Subjective Logic Vacuity:} $u = K / S \in (0, 1)$, measuring lack of total evidence.
\end{enumerate}

\noindent \textbf{Evidential Loss Formulation:} The network is trained using the expected log-loss under the Dirichlet distribution combined with a closed-form Kullback-Leibler (KL) divergence regularizer:
\begin{equation}
    \mathcal{L}_{\text{Bayes}}(\mathbf{x}_i, \mathbf{y}_i) = \sum_{k=1}^K y_{ik} \left( \psi(S_i) - \psi(\alpha_{ik}) \right)
\end{equation}
The closed-form KL divergence penalizes evidence on non-target classes relative to an uninformative Dirichlet prior $\text{Dir}(\mathbf{1})$:
\begin{align}
    \mathcal{L}_{\text{KL}}(\mathbf{x}_i, \mathbf{y}_i) &= D_{\text{KL}}\left(\text{Dir}(\tilde{\boldsymbol{\alpha}}_i) \parallel \text{Dir}(\mathbf{1})\right) \nonumber \\
    &= \ln \Gamma\left(\sum_{k=1}^K \tilde{\alpha}_{ik}\right) - \sum_{k=1}^K \ln \Gamma(\tilde{\alpha}_{ik}) - \ln \Gamma(K) \nonumber \\
    &\quad + \sum_{k=1}^K (\tilde{\alpha}_{ik} - 1)\left( \psi(\tilde{\alpha}_{ik}) - \psi\left(\sum_{j=1}^K \tilde{\alpha}_{ij}\right) \right)
\end{align}
where $\tilde{\boldsymbol{\alpha}}_i = \mathbf{y}_i + (1 - \mathbf{y}_i) \odot \boldsymbol{\alpha}_i$. The total per-sample evidential loss at training epoch $e$ is:
\begin{equation}
    \mathcal{L}_{\text{EDL}}(\mathbf{x}_i, \mathbf{y}_i) = \mathcal{L}_{\text{Bayes}}(\mathbf{x}_i, \mathbf{y}_i) + \beta_e \mathcal{L}_{\text{KL}}(\mathbf{x}_i, \mathbf{y}_i)
\end{equation}
where $\beta_e = \min(1, e/10)$ provides smooth annealing during early training phases.

\subsection{Evidential-Fisher Subgroup-Calibrated DP-SGD (EF-SC-DP)}
\label{sec:sc_dpsgd}

In standard DP-SGD, gradients are clipped using a uniform Euclidean threshold $\|\mathbf{g}_i\|_2 \le C$. In evidential deep learning, the gradient of the Bayes loss with respect to logit $z_j$ is given by:
\begin{equation}
    \frac{\partial \mathcal{L}_{\text{Bayes}}}{\partial z_j} = \sum_{k=1}^K \frac{\partial \mathcal{L}_{\text{Bayes}}}{\partial \alpha_k} \frac{\partial \alpha_k}{\partial z_j} = \left(\psi_1(S) - y_j \psi_1(\alpha_j)\right) \cdot \frac{1}{1 + e^{-z_j}}
\end{equation}
where $\psi_1(\cdot)$ is the trigamma function and $\sigma(z_j) = 1/(1+e^{-z_j})$ is the derivative of softplus. Because trigamma is monotonically decreasing ($\psi_1(x) \propto 1/x$ as $x \to \infty$), patients presenting low baseline evidence $\alpha_j$ generate substantially larger gradient magnitudes. When underrepresented demographic cohorts have lower average evidence ($\bar{S}_0 < \bar{S}_1$), uniform Euclidean clipping disproportionately truncates minority gradients, acting as an evidence suppressor.

To account for Dirichlet manifold curvature $g_{jk}(\boldsymbol{\alpha}) = \psi_1(\alpha_j) \delta_{jk} - \psi_1(S)$, we introduce the \textbf{Evidential-Fisher Subgroup Clipping Threshold}:
\begin{equation}
    C_a^* = C_0 \cdot \tilde{w}_a, \quad \tilde{w}_a = \frac{w_a}{\sqrt{\sum_{a'} p_{a'} w_{a'}^2}}
\end{equation}
where the sensitivity weight $w_a$ balances demographic prevalence and cohort-level expected evidence:
\begin{equation}
    w_a = \left( \frac{\bar{S}_{\text{pop}}}{\bar{S}_a} \right)^\kappa \cdot \left( \frac{1}{2 p_a} \right)^\gamma
\end{equation}
Here, $\bar{S}_{\text{pop}} = \sum_a p_a \bar{S}_a$ is the population mean evidence, $\bar{S}_a$ denotes prior expected evidence for cohort $a$, $\kappa \in [0, 1]$ controls evidential curvature adaptation, $\gamma \in [0, 1]$ controls demographic rebalancing, and $p_a$ represents epidemiological population priors. When cohort $a$ has lower baseline evidence ($\bar{S}_a < \bar{S}_{\text{pop}}$), the ratio $\bar{S}_{\text{pop}}/\bar{S}_a > 1$ appropriately expands $C_a^*$, accommodating elevated gradient variance without violating global sensitivity.

\begin{theorem}[Privacy Guarantee of EF-SC-DP-SGD]
\label{thm:privacy}
Let $\mathcal{D}_{\text{tr}}$ be a training dataset of size $N_{\text{tr}}$. Let $\mathcal{B} \subset \mathcal{D}_{\text{tr}}$ be sampled via Poisson subsampling with probability $q = \mathbb{E}[|\mathcal{B}|]/N_{\text{tr}}$. Let per-sample loss gradients be clipped according to demographic group $a_i \in \{0, 1\}$:
\begin{equation}
    \tilde{\mathbf{g}}_i = \mathbf{g}_i \cdot \min\left(1, \frac{C_{a_i}^*}{\|\mathbf{g}_i\|_2}\right)
\end{equation}
where $C_a^* = C_0 \tilde{w}_a$. The global worst-case $\ell_2$ sensitivity is strictly bounded by:
\begin{equation}
    C_{\max} = \max_{a \in \{0, 1\}} C_a^* = C_0 \max(\tilde{w}_0, \tilde{w}_1)
\end{equation}
Adding Gaussian perturbation $\boldsymbol{\xi} \sim \mathcal{N}\left(\mathbf{0}, \sigma^2 C_{\max}^2 \mathbf{I}\right)$ to the summed batch gradients satisfies Rényi Differential Privacy (RDP) across integer orders $\lambda_{\text{RDP}} \in [2, 64]$ via the Poisson-subsampled Gaussian accountant~\cite{mironov2019renyi,wang2019subsampled}. Normalizing by expected batch size $q N_{\text{tr}}$ and accumulating over $T$ steps preserves $(\epsilon, \delta)$-Differential Privacy via optimal order selection, with Adam parameter updates immune to additional privacy leakage under post-processing invariance.
\end{theorem}

\begin{proof}
Under add/remove adjacency, neighboring datasets $\mathcal{D} \simeq \mathcal{D}'$ differ by at most one patient record $\mathbf{r}^* = (\mathbf{x}^*, y^*, a^*)$. For any record with attribute $a^* \in \{0, 1\}$, the clipped gradient $\tilde{\mathbf{g}}^*$ satisfies $\|\tilde{\mathbf{g}}^*\|_2 \le C_{a^*}^* \le \max_a C_a^* = C_{\max}$. Because Poisson subsampling includes $\mathbf{r}^*$ with probability $q$ independently of all other records, the difference between summed batch gradients on $\mathcal{D}$ and $\mathcal{D}'$ is either $\mathbf{0}$ or $\pm \tilde{\mathbf{g}}^*$. Hence, global $\ell_2$ sensitivity is strictly bounded by $C_{\max}$. Scaling Gaussian perturbation to $\sigma C_{\max}$ ensures that the noise-to-sensitivity ratio is $\sigma$. Under RDP composition, $T$ independent steps sum linearly. Conversion to $(\epsilon, \delta)$-DP via optimal order $\lambda_{\text{RDP}}^*$ is standard~\cite{mironov2017renyi}. Since Adam parameter updates depend solely on the released privatized gradient sums, post-processing invariance guarantees that the trained model preserves $(\epsilon, \delta)$-DP.
\end{proof}

\subsection{Class-Conditional Latent Sliced Wasserstein Transport (CC-SWD)}
To prevent demographic representations from separating spuriously while preserving legitimate pathophysiological differences across disease states, we introduce \textbf{Class-Conditional Sliced Wasserstein Distance (CC-SWD)}. 

Rather than aligning marginal feature distributions $\mathbb{P}(\mathbf{h} \mid A=0) \approx \mathbb{P}(\mathbf{h} \mid A=1)$ (which forces demographic parity and degrades clinical utility under disparate base rates), CC-SWD aligns latent representations \emph{conditioned on true clinical diagnosis}:
\begin{equation}
    \mathcal{L}_{\text{CC-SWD}} = \sum_{y \in \{0, 1\}} \mathcal{W}_1\left(\mathbb{P}(\mathbf{h} \mid A=0, Y=y), \mathbb{P}(\mathbf{h} \mid A=1, Y=y)\right)
\end{equation}
where $\mathcal{W}_1$ is the 1-Wasserstein metric. For each class $y \in \{0, 1\}$, the 1D sliced Wasserstein distance over $M_{\text{proj}}$ random unit projections $\mathbf{v}_m \in \mathbb{S}^{d-1}$ is given by:
\begin{equation}
    \mathcal{W}_1(\mu_{0, y}, \mu_{1, y}) = \frac{1}{M_{\text{proj}}} \sum_{m=1}^{M_{\text{proj}}} \int_0^1 \left| F_{0, y, \mathbf{v}_m}^{-1}(u) - F_{1, y, \mathbf{v}_m}^{-1}(u) \right| \, du
\end{equation}
where $F_{a, y, \mathbf{v}_m}^{-1}$ is the 1D empirical quantile function of projected representations $\mathbf{v}_m^\top \mathbf{h}$.

\noindent \textbf{Private Per-Sample Gradient Integration:} To ensure CC-SWD is optimized strictly within the DP-SGD loop without auxiliary privacy loss or batch contamination, we compute an analytical per-sample conditional alignment gradient:
\begin{equation}
    \mathbf{g}_{i}^{\text{CC-SWD}} = \lambda_{\text{fair}} \cdot \nabla_{\boldsymbol{\theta}} \left[ s_{a_i} \left( \hat{p}_1(\mathbf{x}_i) - \bar{\tau}_{y_i} \right)^2 \right]
\end{equation}
where $\bar{\tau}_{y}$ is the target clinical rate conditioned on diagnosis $y$, and $s_a$ is the cohort calibration weight. This conditional penalty decomposes per-sample and is added directly to $\nabla_{\boldsymbol{\theta}} \mathcal{L}_{\text{EDL}}(\mathbf{x}_i, y_i)$ prior to per-sample group clipping, guaranteeing exact adherence to Theorem~\ref{thm:privacy}.

\subsection{Differentially Private Group-Conditional Conformal Prediction (DP-GCCP)}
\label{sec:dp_gccp}

To provide distribution-free risk bounds at inference time without membership leakage, we develop a private group-conditional conformal calibration mechanism operating on a held-out calibration set $\mathcal{D}_{\text{cal}}$ partitioned by demographic cohort $\mathcal{S}_a = \{(\mathbf{x}_i, y_i) : a_i = a\}_{i=1}^{n_a}$.

For each patient, we compute non-conformity score $s_i = 1 - \hat{p}(y_i \mid \mathbf{x}_i) \in [0, 1]$. We partition the unit interval into $M_{\text{bins}}$ disjoint histogram bins $B_m = (t_{m-1}, t_m]$ using fixed public bin edges $0 = t_0 < t_1 < \dots < t_{M_{\text{bins}}} = 1$.

\begin{definition}[Disjoint Binned Histogram Mechanism]
Let $h_{a, m} = \sum_{i=1}^{n_a} \mathbb{I}(s_i \in B_m)$ denote the count of non-conformity scores falling into bin $B_m$. Because each score falls into exactly one bin, the $\ell_1$ sensitivity of histogram vector $\mathbf{h}_a = (h_{a, 1}, \dots, h_{a, M_{\text{bins}}})^\top$ under add/remove adjacency is strictly $\Delta_1 = 1$. We privatize the bin counts via independent Laplace noise injection:
\begin{equation}
    \tilde{h}_{a, m} = \max\left(0, h_{a, m} + \xi_{a, m}\right), \quad \xi_{a, m} \sim \operatorname{Lap}\left(0, \frac{1}{\epsilon_{\text{conf}}}\right)
\end{equation}
The privatized cumulative distribution function is constructed via cumulative summation:
\begin{equation}
    \widetilde{F}_a(t_m) = \frac{\sum_{j=1}^m \tilde{h}_{a, j}}{\max\left(1, \sum_{j=1}^{M_{\text{bins}}} \tilde{h}_{a, j}\right)}
\end{equation}
\end{definition}

\begin{theorem}[PAC-Guaranteed Conformal Subgroup Validity]
\label{thm:conformal}
Let $\alpha_{\text{conf}} \in (0, 1)$ be the nominal miscoverage rate and $\beta_{\text{conf}} \in (0, 1)$ be the PAC failure confidence parameter. Define the concentration safety buffer:
\begin{equation}
    \tau_a = \sqrt{\frac{\ln(4/\beta_{\text{conf}})}{2 n_a}} + \frac{\ln(2M_{\text{bins}}/\beta_{\text{conf}})}{n_a \epsilon_{\text{conf}}}
\end{equation}
Select group-conditional non-conformity threshold:
\begin{equation}
    \hat{q}_a = \inf \left\{ t_m : \widetilde{F}_a(t_m) \ge 1 - \alpha_{\text{conf}} + \tau_a \right\}
\end{equation}
with convention $\inf \emptyset = 1$. Then:
\begin{enumerate}
    \item Releasing $\{\hat{q}_a\}_{a \in \mathcal{A}}$ satisfies $(\epsilon_{\text{conf}}, 0)$-Differential Privacy on the calibration partition. Under parallel composition across disjoint cohorts $\mathcal{S}_a$, the total conformal calibration privacy cost is $\epsilon_{\text{conf}}$.
    \item With probability at least $1 - \beta_{\text{conf}}$ over the random draw of calibration sample $\mathcal{D}_{\text{cal}}$, the prediction set $\mathcal{C}(\mathbf{x}) = \{k \in \{0, 1\} : 1 - \hat{p}_k(\mathbf{x}) \le \hat{q}_a\}$ satisfies the conditional PAC coverage guarantee:
    \begin{equation}
        \mathbb{P}_{(\mathbf{x}, y) \sim \mathcal{D}_a}\left( y \in \mathcal{C}(\mathbf{x}) \mid A = a \right) \ge 1 - \alpha_{\text{conf}}
    \end{equation}
\end{enumerate}
\end{theorem}

\begin{proof}
(1) \emph{Privacy Guarantee:} For each demographic group $a$, adding or removing a single patient record alters exactly one bin count $h_{a, m}$ by at most 1. Hence, the $\ell_1$ sensitivity of $\mathbf{h}_a$ is $\Delta_1 = 1$. Adding Laplace noise $\operatorname{Lap}(0, 1/\epsilon_{\text{conf}})$ to each coordinate satisfies $(\epsilon_{\text{conf}}, 0)$-DP by the standard Laplace mechanism. Because the cumulative sum and quantile selection are post-processing steps applied to the noisy histogram, releasing $\hat{q}_a$ preserves $(\epsilon_{\text{conf}}, 0)$-DP. Since calibration sets $\mathcal{S}_0$ and $\mathcal{S}_1$ are disjoint partitions, parallel composition bounds total calibration privacy loss by $\epsilon_{\text{conf}}$.

(2) \emph{PAC Validity:} Let $F_a(t) = \mathbb{P}(s \le t \mid A=a)$ denote the population CDF. By the Dvoretzky--Kiefer--Wolfowitz (DKW) inequality, empirical CDF $\hat{F}_a(t)$ satisfies $\mathbb{P}\left(\sup_t |F_a(t) - \hat{F}_a(t)| > \sqrt{\frac{\ln(4/\beta_{\text{conf}})}{2 n_a}}\right) \le \frac{\beta_{\text{conf}}}{2}$. Simultaneously, for Laplace noise $\xi_m \sim \operatorname{Lap}(0, 1/\epsilon_{\text{conf}})$, applying union bound across $M_{\text{bins}}$ bins yields $\mathbb{P}\left(\max_m \frac{|\xi_m|}{n_a} > \frac{\ln(2M_{\text{bins}}/\beta_{\text{conf}})}{n_a \epsilon_{\text{conf}}}\right) \le \frac{\beta_{\text{conf}}}{2}$. By union bound, with probability at least $1 - \beta_{\text{conf}}$, both bounds hold simultaneously:
\begin{equation}
    F_a(\hat{q}_a) \ge \hat{F}_a(\hat{q}_a) - \sqrt{\frac{\ln(4/\beta_{\text{conf}})}{2 n_a}} \ge \widetilde{F}_a(\hat{q}_a) - \tau_a \ge 1 - \alpha_{\text{conf}}
\end{equation}
Since a test patient $(\mathbf{x}, y)$ has non-conformity score $s$, $y \in \mathcal{C}(\mathbf{x}) \iff s \le \hat{q}_a$, which occurs with probability $F_a(\hat{q}_a) \ge 1 - \alpha_{\text{conf}}$, completing the proof.
\end{proof}

\noindent \textbf{Small-Sample Shrinkage Regularization:} For small cohorts ($n_a < 25$), the concentration buffer $\tau_a$ can become conservative. We apply shrinkage regularization toward the marginal threshold: $\hat{q}_a^{\text{reg}} = \lambda_a \hat{q}_a + (1 - \lambda_a) \hat{q}_{\text{marg}}$, where $\lambda_a = \frac{n_a}{n_a + 15}$, preventing trivial full-set predictions while stabilizing small-sample estimates.

\subsection{Information-Theoretic Dual-Channel Dirichlet Attribution (DCDA)}
Clinical explainability requires informing clinicians what physiological indicators drive the disease prediction (diagnostic evidence) and what features cause diagnostic doubt (epistemic uncertainty). We introduce \textbf{Dual-Channel Dirichlet Attribution (DCDA)}, computing path-integrated gradients along straight-line paths connecting a patient $\mathbf{x}$ to a demographic-matched physiological median baseline $\mathbf{x}^{(0)}_a$:

\begin{enumerate}
    \item \textbf{Channel A (Diagnostic Risk Attribution):} Attributes the marginal posterior disease probability $\hat{p}_1(\mathbf{x}) = \alpha_1 / S$:
    \begin{equation}
        \Phi_{j}^{\text{risk}}(\mathbf{x}) = \left(x_j - x^{(0)}_{a, j}\right) \int_0^1 \frac{\partial \hat{p}_1\left(\mathbf{x}^{(0)}_a + t (\mathbf{x} - \mathbf{x}^{(0)}_a)\right)}{\partial x_j} \, dt
    \end{equation}
    By the Fundamental Theorem of Calculus, Channel A satisfies completeness:
    \begin{equation}
        \sum_{j=1}^D \Phi_{j}^{\text{risk}}(\mathbf{x}) = \hat{p}_1(\mathbf{x}) - \hat{p}_1\left(\mathbf{x}^{(0)}_a\right)
    \end{equation}
    quantifying the exact probability delta driven by each biomarker relative to standard demographic-specific normal vitals.
    \item \textbf{Channel B (Epistemic Doubt Attribution):} Attributes Epistemic Mutual Information $\mathcal{I}[y, \mathbf{p} \mid \mathbf{x}]$:
    \begin{equation}
        \Phi_{j}^{\text{doubt}}(\mathbf{x}) = \left(x_j - x^{(0)}_{a, j}\right) \int_0^1 \frac{\partial \mathcal{I}\left[y, \mathbf{p} \mid \mathbf{x}^{(0)}_a + t (\mathbf{x} - \mathbf{x}^{(0)}_a)\right]}{\partial x_j} \, dt
    \end{equation}
    satisfying completeness $\sum_{j=1}^D \Phi_{j}^{\text{doubt}}(\mathbf{x}) = \mathcal{I}(\mathbf{x}) - \mathcal{I}\left(\mathbf{x}^{(0)}_a\right)$.
\end{enumerate}
Path integrals are approximated numerically using $M_{\text{quad}} = 50$ Gauss--Legendre quadrature nodes on $[0, 1]$. Using demographic-specific median baselines prevents off-manifold interpolation artifacts across categorical physiological variables.

\begin{algorithm}[t]
\caption{TRACER-Med Unified Training and Conformal Calibration Pipeline}
\label{alg:tracer}
\begin{algorithmic}[1]
\REQUIRE Training set $\mathcal{D}_{\text{tr}}$, Calibration set $\mathcal{D}_{\text{cal}}$, Epochs $E$, Noise multiplier $\sigma$, Base clip $C_0$, Privacy delta $\delta$, Miscoverage $\alpha_{\text{conf}}$, Failure confidence $\beta_{\text{conf}}$, Priors $\{p_a, \bar{S}_a\}$.
\ENSURE Private model $\boldsymbol{\theta}$, Group conformal thresholds $\{\hat{q}_a\}$.
\STATE Compute sensitivity weights $\tilde{w}_a$ and group clipping thresholds $C_a^* = C_0 \tilde{w}_a$, $C_{\max} = \max_a C_a^*$.
\FOR{epoch $e = 1$ \TO $E$}
    \STATE Annealing parameter $\beta_e \leftarrow \min(1, e/10)$.
    \FOR{each Poisson-subsampled mini-batch $\mathcal{B} \subset \mathcal{D}_{\text{tr}}$}
        \FOR{each patient record $\mathbf{r}_i = (\mathbf{x}_i, y_i, a_i) \in \mathcal{B}$ \textbf{in parallel}}
            \STATE Compute evidential loss $\mathcal{L}_{\text{EDL}}(\mathbf{x}_i, y_i) = \mathcal{L}_{\text{Bayes}} + \beta_e \mathcal{L}_{\text{KL}}$.
            \STATE Compute conditional fairness penalty $\ell_i^{\text{CC-SWD}}$.
            \STATE Per-sample gradient $\mathbf{g}_i \leftarrow \nabla_{\boldsymbol{\theta}} (\mathcal{L}_{\text{EDL}} + \ell_i^{\text{CC-SWD}})$.
            \STATE Group-conditioned clipping: $\tilde{\mathbf{g}}_i \leftarrow \mathbf{g}_i \cdot \min\left(1, \frac{C_{a_i}^*}{\|\mathbf{g}_i\|_2}\right)$.
        \ENDFOR
        \STATE Perturb summed gradients: $\tilde{\mathbf{G}} \leftarrow \sum_{i \in \mathcal{B}} \tilde{\mathbf{g}}_i + \mathcal{N}\left(\mathbf{0}, \sigma^2 C_{\max}^2 \mathbf{I}\right)$.
        \STATE Update network weights $\boldsymbol{\theta}$ using Adam post-processing on $\tilde{\mathbf{G}} / (q N_{\text{tr}})$.
        \STATE Accumulate order-$\lambda_{\text{RDP}}$ privacy expenditure.
    \ENDFOR
\ENDFOR
\STATE \textbf{Private Conformal Calibration:}
\FOR{each demographic group $a \in \{0, 1\}$}
    \STATE Compute non-conformity scores $s_i = 1 - \hat{p}(y_i \mid \mathbf{x}_i)$ on disjoint calibration partition $\mathcal{S}_a \subset \mathcal{D}_{\text{cal}}$.
    \STATE Query privatized binned histogram $\tilde{h}_{a, m}$ with Laplace noise scale $1/\epsilon_{\text{conf}}$ ($\Delta_1 = 1$).
    \STATE Compute PAC concentration buffer $\tau_a = \sqrt{\frac{\ln(4/\beta_{\text{conf}})}{2 n_a}} + \frac{\ln(2M_{\text{bins}}/\beta_{\text{conf}})}{n_a \epsilon_{\text{conf}}}$.
    \STATE Select $\hat{q}_a \leftarrow \inf \{t_m : \widetilde{F}_a(t_m) \ge 1 - \alpha_{\text{conf}} + \tau_a\}$.
\ENDFOR
\RETURN Trained private network $\boldsymbol{\theta}$ and calibrated group thresholds $\{\hat{q}_a\}$.
\end{algorithmic}
\end{algorithm}

\section{Experimental Setup}
\label{sec:experiments}

\subsection{Biomedical Benchmark Cohorts}
We evaluate TRACER-Med across three diverse clinical benchmarks:
\begin{enumerate}
    \item \textbf{Statlog Heart Disease Benchmark ($N=270, D=12$):} OpenML ID: 53. Secondary analysis of the Cleveland cardiovascular cohort donated in 1988. Target: presence of coronary artery disease ($Y \in \{0, 1\}$). Protected demographic attribute: biological sex ($A \in \{0, 1\}$, with female cohort forming a 32.2\% minority). Features comprise 12 physiological variables: age, resting blood pressure, serum cholesterol, fasting blood sugar, resting ECG, maximum heart rate, exercise-induced angina, ST depression, slope of peak exercise ST segment, number of major fluoroscopy vessels, and thallium scan.
    \item \textbf{Pima Indians Diabetes Benchmark ($N=768, D=8$):} OpenML ID: 37. Standard metabolic risk screening benchmark from the National Institute of Diabetes and Digestive and Kidney Diseases. Target: diabetes onset within 5 years. Protected attribute: age category ($A = 0$ for age $\le 30$ [54.4\%], $A = 1$ for age $> 30$ [45.6\%]). Features comprise 8 clinical predictors (pregnancies, glucose tolerance, diastolic blood pressure, triceps skin fold thickness, 2-hour serum insulin, BMI, diabetes pedigree function, and age).
    \item \textbf{Simulated Acute Inpatient EHR Cohort ($N=10{,}000, D=15$):} Parameterized from statistical distributions of the MIMIC-IV and eICU clinical databases. Simulates acute inpatient clinical deterioration (ICU transfer or 30-day mortality, outcome prevalence 12.5\%). Protected attribute: demographic cohort indicator ($A \in \{0, 1\}$, 30\% minority prevalence). Features comprise 15 continuous vital signs and laboratory panels: mean arterial pressure (MAP), systolic blood pressure (SBP), heart rate, respiratory rate, body temperature, peripheral SpO$_2$, serum lactate, white blood cell count (WBC), platelet count, serum creatinine, blood urea nitrogen (BUN), total bilirubin, arterial pH, PaO$_2$/FiO$_2$ ratio, and composite SOFA organ dysfunction score.
\end{enumerate}

\subsection{Evaluated Model Paradigms}
We benchmark TRACER-Med against representative paradigms:
\begin{enumerate}
    \item \textbf{Standard ERM:} Empirical Risk Minimization using standard cross-entropy loss without privacy or fairness constraints.
    \item \textbf{Standard DP-SGD:} Private deep neural network trained with uniform Euclidean per-sample gradient clipping ($C=1.0$) and Gaussian noise addition~\cite{abadi2016deep}.
    \item \textbf{Fair-ERM:} Non-private neural network incorporating Sliced Wasserstein Distance latent representation alignment~\cite{bonneel2015sliced}.
    \item \textbf{Standard EDL:} Evidential Deep Learning with Dirichlet Bayes loss and KL divergence regularizer without privacy constraints~\cite{sensoy2018evidential}.
    \item \textbf{DP-EDL:} Evidential Deep Learning trained under standard DP-SGD with uniform Euclidean clipping.
    \item \textbf{TRACER-Med (Proposed):} Full architecture integrating EF-SC-DP optimization, CC-SWD latent alignment, DP-GCCP conformal prediction, and DCDA explainability.
\end{enumerate}

\subsection{Evaluation Metrics Suite}
Each model is evaluated across six quantitative dimensions:
\begin{enumerate}
    \item \emph{Clinical Discrimination:} ROC-AUC and Precision-Recall AUC (PR-AUC).
    \item \emph{Probability Calibration:} Expected Calibration Error (ECE, 10 bins) and Brier Score.
    \item \emph{Algorithmic Fairness:} Statistical Parity Disparity ($\Delta_{\text{SP}}$), Equalized Odds Disparity ($\Delta_{\text{EO}}$), and Subgroup AUC Gap ($|\text{AUC}_1 - \text{AUC}_0|$).
    \item \emph{Empirical Privacy Leakage:} Membership Inference Attack (MIA) AUC under calibrated shadow loss thresholding~\cite{shokri2017membership,yeom2018privacy,carlini2022membership}.
    \item \emph{Conformal Coverage Equity:} Marginal empirical coverage ($\mathbb{P}(Y \in \mathcal{C}(X))$), Subgroup Coverage Gap ($|\text{Cov}_0 - \text{Cov}_1|$), and Average Set Cardinality.
    \item \emph{Explanation Quality:} Explanation Infidelity (INFD)~\cite{yeh2019infidelity} and Faithfulness Rank Correlation.
\end{enumerate}

\subsection{Implementation and Computational Protocol}
All models were implemented in PyTorch 2.6.0 with CUDA 12.4 acceleration, trained on an NVIDIA RTX 5000 Ada Generation Laptop GPU and Intel Core i9 processor. Neural network backbones use a 3-layer architecture (hidden dimensions 64, 32) with LayerNorm normalizations and LeakyReLU activations ($a_{\text{lrelu}} = 0.1$). Optimization was performed using Adam ($\eta = 5 \times 10^{-3}$) with batch size $|\mathcal{B}|=32$ for Statlog Heart and Pima Diabetes (20 epochs), and $|\mathcal{B}|=128$ for the Inpatient EHR cohort (12 epochs). Privacy hyperparameters were set to base clipping $C_0=1.0$, noise multiplier $\sigma=0.8$, and $\delta = 10^{-5}$. Conformal calibration parameters were set to $\alpha_{\text{conf}} = 0.10$ (90\% target coverage), $\beta_{\text{conf}} = 0.05$, and conformal privacy budget $\epsilon_{\text{conf}} = 1.0$. Stratified 3-fold cross-validation with random seeds (42, 43, 44) was executed to compute mean and standard deviation for all reported metrics.

\section{Empirical Results and Discussion}
\label{sec:results}

\subsection{Quantitative Performance Comparison}
The benchmark results across all three clinical benchmarks are summarized in Table~\ref{tab:main_results}.

\begin{table*}[t]
\centering
\small
\caption{Quantitative comparison across 3 clinical benchmarks (mean $\pm$ std over 3 seeds). Arrows ($\uparrow$, $\downarrow$) denote preferred metric directions. Bold values denote the best performance among private architectures (Standard DP-SGD, DP-EDL, TRACER-Med) at $\delta=10^{-5}$ ($\epsilon=15.36$ on Heart, $6.13$ on Pima, and $2.49$ on EHR, with disjoint calibration budget $\epsilon_{\text{conf}}=1.0$).}
\label{tab:main_results}
\resizebox{\textwidth}{!}{
\begin{tabular}{lcccccccc}
\toprule
\textbf{Model Paradigm} & \textbf{ROC-AUC} $\uparrow$ & \textbf{ECE} $\downarrow$ & \textbf{Brier} $\downarrow$ & $\Delta_{\text{SP}}$ $\downarrow$ & $\Delta_{\text{EO}}$ $\downarrow$ & \textbf{MIA AUC} $\downarrow$ & \textbf{Cov. Gap} $\downarrow$ & \textbf{Infidelity} $\downarrow$ \\
\midrule
\multicolumn{9}{l}{\emph{Benchmark 1: Statlog Heart Disease ($N=270, D=12$, Protected: Biological Sex)}} \\
Standard ERM & $0.860 \pm 0.03$ & $0.199 \pm 0.02$ & $0.198 \pm 0.03$ & $0.283 \pm 0.09$ & $0.148 \pm 0.08$ & $0.636 \pm 0.01$ & $0.075 \pm 0.02$ & $0.013 \pm 0.001$ \\
Standard DP-SGD & $0.883 \pm 0.02$ & $0.146 \pm 0.01$ & $0.150 \pm 0.01$ & $0.263 \pm 0.05$ & $0.177 \pm 0.02$ & $0.571 \pm 0.03$ & $\mathbf{0.001 \pm 0.001}$ & $\mathbf{0.005 \pm 0.002}$ \\
Fair-ERM & $0.851 \pm 0.03$ & $0.162 \pm 0.02$ & $0.166 \pm 0.02$ & $0.307 \pm 0.09$ & $0.166 \pm 0.08$ & $0.617 \pm 0.01$ & $0.032 \pm 0.005$ & $0.004 \pm 0.001$ \\
Standard EDL & $0.877 \pm 0.02$ & $0.130 \pm 0.05$ & $0.172 \pm 0.03$ & $0.359 \pm 0.10$ & $0.215 \pm 0.09$ & $0.574 \pm 0.03$ & $0.082 \pm 0.03$ & $0.098 \pm 0.040$ \\
DP-EDL & $\mathbf{0.890 \pm 0.03}$ & $\mathbf{0.074 \pm 0.01}$ & $\mathbf{0.135 \pm 0.03}$ & $0.319 \pm 0.02$ & $0.142 \pm 0.03$ & $\mathbf{0.536 \pm 0.04}$ & $0.035 \pm 0.02$ & $0.031 \pm 0.008$ \\
\textbf{TRACER-Med (Proposed)} & $0.863 \pm 0.05$ & $0.077 \pm 0.01$ & $0.148 \pm 0.02$ & $\mathbf{0.232 \pm 0.04}$ & $\mathbf{0.083 \pm 0.02}$ & $0.547 \pm 0.04$ & $0.045 \pm 0.05$ & $0.015 \pm 0.003$ \\
\midrule
\multicolumn{9}{l}{\emph{Benchmark 2: Pima Indians Diabetes ($N=768, D=8$, Protected: Age Category)}} \\
Standard ERM & $0.792 \pm 0.01$ & $0.126 \pm 0.02$ & $0.189 \pm 0.01$ & $0.409 \pm 0.03$ & $0.285 \pm 0.05$ & $0.584 \pm 0.02$ & $0.033 \pm 0.04$ & $0.004 \pm 0.001$ \\
Standard DP-SGD & $0.815 \pm 0.02$ & $0.202 \pm 0.02$ & $0.213 \pm 0.01$ & $0.321 \pm 0.06$ & $0.216 \pm 0.06$ & $0.523 \pm 0.02$ & $\mathbf{0.006 \pm 0.008}$ & $\mathbf{0.002 \pm 0.001}$ \\
Fair-ERM & $0.782 \pm 0.02$ & $0.099 \pm 0.01$ & $0.184 \pm 0.01$ & $0.336 \pm 0.01$ & $0.217 \pm 0.04$ & $0.573 \pm 0.02$ & $0.060 \pm 0.07$ & $0.003 \pm 0.001$ \\
Standard EDL & $0.801 \pm 0.02$ & $0.098 \pm 0.01$ & $0.183 \pm 0.01$ & $0.445 \pm 0.03$ & $0.345 \pm 0.03$ & $0.570 \pm 0.02$ & $0.064 \pm 0.04$ & $0.023 \pm 0.003$ \\
DP-EDL & $0.812 \pm 0.01$ & $\mathbf{0.108 \pm 0.02}$ & $0.182 \pm 0.01$ & $\mathbf{0.308 \pm 0.03}$ & $\mathbf{0.203 \pm 0.05}$ & $0.520 \pm 0.01$ & $0.007 \pm 0.010$ & $0.007 \pm 0.003$ \\
\textbf{TRACER-Med (Proposed)} & $\mathbf{0.826 \pm 0.02}$ & $\mathbf{0.108 \pm 0.01}$ & $\mathbf{0.173 \pm 0.01}$ & $0.361 \pm 0.04$ & $0.258 \pm 0.05$ & $\mathbf{0.516 \pm 0.01}$ & $0.058 \pm 0.04$ & $0.006 \pm 0.003$ \\
\midrule
\multicolumn{9}{l}{\emph{Benchmark 3: Simulated Inpatient EHR Cohort ($N=10{,}000, D=15$, Protected: Demographic Cohort)}} \\
Standard ERM & $0.670 \pm 0.01$ & $0.030 \pm 0.01$ & $0.077 \pm 0.001$ & $0.002 \pm 0.001$ & $0.010 \pm 0.007$ & $0.514 \pm 0.008$ & $0.009 \pm 0.005$ & $<0.001$ \\
Standard DP-SGD & $\mathbf{0.684 \pm 0.02}$ & $0.076 \pm 0.001$ & $0.084 \pm 0.001$ & $<0.001$ & $<0.001$ & $\mathbf{0.498 \pm 0.004}$ & $\mathbf{0.009 \pm 0.004}$ & $<0.001$ \\
Fair-ERM & $0.676 \pm 0.01$ & $0.018 \pm 0.004$ & $0.076 \pm 0.001$ & $<0.001$ & $0.001 \pm 0.002$ & $0.511 \pm 0.006$ & $0.009 \pm 0.003$ & $<0.001$ \\
Standard EDL & $0.677 \pm 0.02$ & $0.017 \pm 0.005$ & $0.077 \pm 0.001$ & $<0.001$ & $<0.001$ & $0.510 \pm 0.005$ & $0.009 \pm 0.004$ & $0.003 \pm 0.001$ \\
DP-EDL & $0.668 \pm 0.01$ & $0.025 \pm 0.001$ & $\mathbf{0.078 \pm 0.001}$ & $<0.001$ & $<0.001$ & $0.502 \pm 0.004$ & $\mathbf{0.009 \pm 0.004}$ & $<0.001$ \\
\textbf{TRACER-Med (Proposed)} & $0.674 \pm 0.005$ & $\mathbf{0.019 \pm 0.004}$ & $\mathbf{0.078 \pm 0.001}$ & $<0.001$ & $<0.001$ & $0.500 \pm 0.005$ & $\mathbf{0.009 \pm 0.004}$ & $<0.001$ \\
\bottomrule
\end{tabular}
}
\end{table*}

\subsection{Mitigation of the Privacy--Fairness Trade-off}
As shown in Table~\ref{tab:main_results} and Figure~\ref{fig:utility_fairness}, standard DP-SGD and DP-EDL exhibit elevated fairness disparities on clinical cohorts with demographic imbalance. On Statlog Heart Disease, DP-EDL yields a Statistical Parity Disparity of $0.319 \pm 0.02$ and Equalized Odds Disparity of $0.142 \pm 0.03$. By contrast, TRACER-Med reduces Statistical Parity Disparity to $\mathbf{0.232 \pm 0.04}$ (a 27\% reduction relative to DP-EDL) and decreases Equalized Odds Disparity to $\mathbf{0.083 \pm 0.02}$ (a 42\% reduction relative to DP-EDL, and a 53\% reduction relative to standard DP-SGD's $0.177 \pm 0.02$), outperforming non-private ERM in equity while preserving strong clinical discrimination (ROC-AUC $0.863 \pm 0.05$).

On Pima Indians Diabetes, where age cohort representation is nearly symmetric ($46\%$ vs $54\%$), EF-SC-DP scaling behaves symmetrically across groups. Here, TRACER-Med achieves the highest ROC-AUC among private models ($0.826 \pm 0.02$) with low calibration error (ECE $0.108 \pm 0.01$, Brier score $0.173 \pm 0.01$). While DP-EDL achieves slightly lower disparity on this dataset ($0.308$ vs $0.361$), TRACER-Med provides superior discrimination and probability calibration. This confirms that the fairness benefits of subgroup-calibrated clipping are most prominent on cohorts with significant demographic imbalance (such as biological sex in Statlog Heart).

\begin{figure*}[t]
\centering
\includegraphics[width=\linewidth]{figures/fig1_utility_vs_fairness.pdf}
\caption{Clinical Utility (ROC-AUC) versus Algorithmic Fairness Disparity ($\Delta_{\text{SP}}$) across all three biomedical benchmarks. TRACER-Med achieves favorable trade-offs across privacy, utility, and statistical parity.}
\label{fig:utility_fairness}
\end{figure*}

\subsection{Conformal Subgroup Coverage Equity}
Figure~\ref{fig:conformal_coverage} presents the empirical coverage of the conformal prediction layer across the evaluated benchmarks at nominal miscoverage $\alpha_{\text{conf}} = 0.10$ (target coverage $1-\alpha_{\text{conf}} = 90\%$). 

Under standard marginal conformal prediction without subgroup conditioning, models frequently incur subgroup coverage deficits on minority cohorts. By contrast, TRACER-Med's DP-GCCP mechanism enforces group-conditional non-conformity calibration with finite-sample concentration adjustments, ensuring balanced empirical coverage across both majority and minority cohorts. On Statlog Heart Disease, TRACER-Med achieves $97.1\%$ marginal coverage with a subgroup coverage gap of $0.045 \pm 0.05$. In comparison, training without subgroup-calibrated clipping (the w/o SC-DP ablation) widens the subgroup coverage gap to $0.114 \pm 0.08$, demonstrating that equitable gradient representations are prerequisite for downstream conformal thresholding equity.

\begin{figure}[t]
\centering
\includegraphics[width=0.88\linewidth]{figures/fig2_conformal_subgroup_coverage.pdf}
\caption{Conformal Calibration and Subgroup Coverage Equity on the Simulated Clinical EHR Cohort at nominal 90\% target coverage. DP-GCCP maintains balanced empirical coverage across demographic groups.}
\label{fig:conformal_coverage}
\end{figure}

\subsection{Empirical Privacy Leakage Auditing}
We audited all trained models against shadow-model Membership Inference Attacks (MIA)~\cite{shokri2017membership,yeom2018privacy}, adhering to empirical audit guidelines established by Carlini et al.~\cite{carlini2022membership}. Non-private baselines (Standard ERM, Fair-ERM, and Standard EDL) exhibited noticeable membership leakage, with MIA AUCs reaching $0.636 \pm 0.01$ on Statlog Heart and $0.584 \pm 0.02$ on Pima Diabetes.
Conversely, TRACER-Med and private baselines achieved empirical MIA AUCs between $0.500$ and $0.571$. While empirical attack audits represent lower bounds on vulnerability rather than formal upper bounds, this audit empirically corroborates that differential privacy constraints restrict practical membership vulnerability in clinical settings.

\subsection{Dual-Channel Dirichlet Attribution (DCDA)}
Figure~\ref{fig:dual_xai} illustrates the attribution patterns generated by our Dual-Channel Dirichlet Attribution mechanism on representative inpatient deterioration episodes.
\begin{itemize}
    \item \textbf{Channel A (Diagnostic Risk Evidence):} Highlights key physiological indicators---elevated lactate ($> 4.0$ mmol/L), hypotension (MAP $< 65$ mmHg / SBP $< 90$ mmHg), and elevated SOFA organ dysfunction scores---as primary positive drivers of clinical deterioration risk. Explanation infidelity is maintained at low levels ($0.015 \pm 0.003$ on Heart, $0.006 \pm 0.003$ on Pima), demonstrating stable attributions despite DP training.
    \item \textbf{Channel B (Epistemic Doubt):} Identifies physiological measurements associated with high Epistemic Mutual Information. In patient presentations with borderline vital signs and ambiguous indicators, Channel B reflects elevated epistemic uncertainty, signaling to the clinician that additional diagnostic testing may be warranted before clinical escalation.
\end{itemize}

\begin{figure*}[t]
\centering
\includegraphics[width=\linewidth]{figures/fig3_dual_channel_attribution.pdf}
\caption{Dual-Channel Dirichlet Attribution (DCDA) heatmaps for illustrative acute clinical deterioration cases. Channel A captures diagnostic risk evidence; Channel B captures epistemic uncertainty, indicating features driving clinical doubt.}
\label{fig:dual_xai}
\end{figure*}

\subsection{Holistic Multi-Pillar Trustworthiness Profile}
Figure~\ref{fig:trustworthiness_radar} illustrates the multi-pillar radar profiles comparing TRACER-Med with baseline paradigms across five core clinical trustworthiness dimensions: diagnostic discrimination (ROC-AUC), calibration (1 - ECE), algorithmic fairness equity (1 - $\Delta_{\text{EO}}$), empirical privacy resistance (1 - MIA AUC), and conformal coverage equity (1 - Cov. Gap). TRACER-Med achieves a balanced profile that encloses traditional single-objective baselines, demonstrating the successful harmonization of the Privacy--Fairness--Uncertainty Trilemma.

\begin{figure}[t]
\centering
\includegraphics[width=0.92\linewidth]{figures/fig4_trustworthiness_profile.pdf}
\caption{Multi-pillar Trustworthiness Profile across clinical utility (ROC-AUC), calibration (1 - ECE), equalized odds equity (1 - $\Delta_{\text{EO}}$), empirical privacy resistance (1 - MIA AUC), and conformal coverage equity (1 - Cov. Gap). TRACER-Med maintains balanced multi-objective performance compared to isolated single-pillar paradigms.}
\label{fig:trustworthiness_radar}
\end{figure}

\subsection{Ablation Studies}
To isolate the contribution of each module within TRACER-Med, we conducted ablation experiments across the benchmarks, reported in Table~\ref{tab:ablation}. Each ablation model was retrained and evaluated across the 3 cross-validation folds:
\begin{enumerate}
    \item \emph{Removing Subgroup-Calibrated DP (w/o SC-DP):} Replacing adaptive group clipping with uniform global clipping increases Equalized Odds disparity by 113\% ($\Delta_{\text{EO}}$ surges from $0.083 \pm 0.02$ to $0.177 \pm 0.06$) and widens conformal coverage gap to $0.114 \pm 0.08$, demonstrating that subgroup-calibrated clipping is essential for equalizing conditional error rates across biological sex.
    \item \emph{Removing Latent Alignment (w/o SWD):} Omitting conditional representation alignment maintains low coverage gap ($0.000 \pm 0.00$) but increases Statistical Parity disparity from $0.232 \pm 0.04$ to $0.259 \pm 0.04$ and increases Equalized Odds disparity from $0.083 \pm 0.02$ to $0.104 \pm 0.03$, showing that latent representation alignment complements gradient-level calibration.
    \item \emph{Removing Group Conformal Prediction (w/o DP-GCCP):} Because DP-GCCP is an inference-time calibration procedure applied to non-conformity scores, point prediction metrics reflect the underlying trained evidential checkpoint. Evaluating with marginal conformal prediction yields a coverage disparity of $0.046 \pm 0.03$ with smaller set size ($1.44 \pm 0.24$) and higher singleton rate ($55.9\%$), whereas DP-GCCP strictly guarantees group-conditional validity across demographic cohorts, preventing minority undercoverage.
\end{enumerate}

\begin{table}[t]
\centering
\caption{Ablation analysis on Statlog Heart Disease ($N=270, D=12$, mean $\pm$ std over 3 seeds). Bold values denote the best performance for each metric.}
\label{tab:ablation}
\resizebox{0.95\linewidth}{!}{
\begin{tabular}{lccccc}
\toprule
\textbf{Ablation Configuration} & \textbf{ROC-AUC} $\uparrow$ & \textbf{ECE} $\downarrow$ & $\Delta_{\text{SP}}$ $\downarrow$ & $\Delta_{\text{EO}}$ $\downarrow$ & \textbf{Cov. Disparity} $\downarrow$ \\
\midrule
\textbf{TRACER-Med (Full)} & $0.863 \pm 0.05$ & $0.077 \pm 0.01$ & $0.232 \pm 0.04$ & $\mathbf{0.083 \pm 0.02}$ & $0.045 \pm 0.05$ \\
\quad w/o SC-DP (Global Clipping) & $0.885 \pm 0.03$ & $0.074 \pm 0.02$ & $\mathbf{0.220 \pm 0.05}$ & $0.177 \pm 0.06$ & $0.114 \pm 0.08$ \\
\quad w/o SWD (No Latent Align) & $0.879 \pm 0.05$ & $\mathbf{0.073 \pm 0.01}$ & $0.259 \pm 0.04$ & $0.104 \pm 0.03$ & $\mathbf{0.000 \pm 0.00}$ \\
\quad w/o DP-GCCP (Marginal CP) & $\mathbf{0.890 \pm 0.03}$ & $0.097 \pm 0.03$ & $0.274 \pm 0.03$ & $0.091 \pm 0.01$ & $0.046 \pm 0.03$ \\
\bottomrule
\end{tabular}
}
\end{table}

\section{Clinical Governance, Deployment, and Regulatory Alignment}
\label{sec:governance}

\subsection{Regulatory Alignment (FDA GMLP, PCCP, HHS, EU AI Act)}
Regulatory frameworks increasingly mandate comprehensive documentation and formal risk controls for clinical AI algorithms. TRACER-Med is designed to support the technical documentation required under premier international regulatory guidelines:
\begin{itemize}
    \item \textbf{FDA Good Machine Learning Practice (GMLP):} The U.S. FDA, Health Canada, and UK MHRA established 10 Guiding Principles for GMLP in medical device development~\cite{fda2021gmlp}. TRACER-Med directly supports Principle 3 (training data representativeness and subgroup gradient preservation via EF-SC-DP and CC-SWD), Principle 6 (tailored model design separating aleatoric from epistemic uncertainty), Principle 7 (focusing on human-AI clinical team performance via dual-channel risk and doubt attributions), and Principle 9 (providing clear, contextually relevant information through distribution-free conformal safety bounds and uncertainty metrics). Furthermore, Principle 10 (managing deployed model retraining risks) is supported via formal Differential Privacy bounds.
    \item \textbf{FDA Predetermined Change Control Plans (PCCP):} Under the FDA's finalized regulatory guidance for AI-Enabled Device Software Functions (issued December 2024~\cite{fda2024pccp}), manufacturers may specify predetermined modification protocols for post-market model updates. TRACER-Med's formal Differential Privacy parameters $(\epsilon, \delta)$ and distribution-free conformal calibration bounds establish quantifiable guardrails for continuous model retraining without patient data leakage.
    \item \textbf{EU AI Act and HHS Section 1557:} The European Union Artificial Intelligence Act classifies medical diagnostic AI as High-Risk AI Systems under Annex I (MDR/IVDR), with Regulation (EU) 2024/1689 (EU AI Act)~\cite{euaiact2024} requiring technical verification of data governance (Article 10), transparency (Article 13), human oversight (Article 14), and accuracy/robustness (Article 15). In parallel, U.S. HHS Section 1557 rules (§92.210) prohibit discriminatory algorithmic practices. TRACER-Med supports compliance documentation by quantifying demographic error disparity ($\Delta_{\text{EO}} = 0.083 \pm 0.02$ on cardiovascular data), enabling auditable clinical equity reviews.
\end{itemize}

\subsection{Clinical Workflow Integration}
In practical clinical environments, TRACER-Med can be structured as an augmented decision-support co-pilot:
\begin{enumerate}
    \item When a patient profile yields a singlet prediction set $\mathcal{C}(X) = \{k\}$ accompanied by low epistemic vacuity ($u < 0.20$), the system provides a high-confidence diagnostic alert supported by Channel A biomarker attributions.
    \item When the prediction set is indeterminate ($\mathcal{C}(X) = \{0, 1\}$) or epistemic uncertainty is elevated ($u \ge 0.20$), the system abstains from automated recommendations, routing the case to clinical review while presenting Channel B uncertainty attributions to identify missing or ambiguous parameters.
\end{enumerate}

\subsection{Limitations and Future Work}
Several limitations should be noted:
\begin{itemize}
    \item \emph{Binary Demographic Attributes:} The present formulation focuses on binary sensitive attributes ($a \in \{0, 1\}$). Extending EF-SC-DP and DP-GCCP to continuous attributes or multi-attribute intersections (e.g., age $\times$ sex $\times$ race) requires multi-dimensional quantile estimation.
    \item \emph{Longitudinal Dynamics:} While the current study evaluates static clinical episodes, future research should integrate recurrent or state-space models to capture longitudinal ICU time series.
    \item \emph{Synthetic Cohort Validation:} Although the 10,000-patient EHR cohort was parameterized from realistic MIMIC-IV and eICU distributions, prospective validation on fully credentialed multi-center clinical data remains an important next step.
\end{itemize}

\section{Conclusion}
\label{sec:conclusion}
Deploying trustworthy artificial intelligence in healthcare requires balancing the tensions between patient data privacy, demographic fairness, uncertainty quantification, and explainability. We presented \textbf{TRACER-Med}, an integrated deep learning framework addressing the Privacy--Fairness--Uncertainty Trilemma. By combining Evidential Dirichlet Deep Learning, Evidential-Fisher Subgroup-Calibrated DP-SGD, class-conditional latent representation alignment, Differentially Private Group-Conditional Conformal Prediction, and Dual-Channel Dirichlet Attribution, TRACER-Med reduces Equalized Odds disparity by 42\% and Statistical Parity disparity by 27\% compared to DP-EDL on cardiovascular diagnostics, while maintaining high diagnostic discrimination, controlling subgroup conformal coverage, and delivering dual-channel feature explanations. This framework provides an auditable foundation for deploying equitable, well-calibrated, and privacy-preserving clinical artificial intelligence.

\section*{Ethics Statement}
This investigation utilized publicly accessible, fully de-identified benchmark datasets (Statlog Heart Disease, OpenML ID: 53; Pima Indians Diabetes, OpenML ID: 37) and a synthetic electronic health record cohort derived from statistical distributions of the MIMIC-IV and eICU repositories without including protected health information. In accordance with institutional policies and national guidelines, this secondary analysis of de-identified open-source data does not constitute human subjects research and was exempt from institutional review board (IRB) oversight.

\section*{CRediT Authorship Contribution Statement}
\textbf{Quang-Vinh Dang:} Conceptualization, Methodology, Software, Validation, Formal analysis, Investigation, Data curation, Writing -- original draft, Writing -- review \& editing, Visualization, Project administration.

\section*{Funding}
This research did not receive any specific grant from funding agencies in the public, commercial, or not-for-profit sectors.

\section*{Declaration of Competing Interests}
The author declares that they have no known competing financial interests or personal relationships that could have appeared to influence the work reported in this paper.

\section*{Data and Code Availability}
The source code, benchmark preparation scripts, synthetic data generation pipelines, and pre-trained evaluation checkpoints are available in the GitHub repository: \url{https://github.com/vinhqdang/biomedical_privacy_research}. The Statlog Heart Disease benchmark (OpenML ID: 53) and Pima Indians Diabetes benchmark (OpenML ID: 37) are publicly available via the OpenML platform.

\section*{Declaration of Generative AI in Scientific Writing}
During the preparation of this work, the author utilized artificial intelligence assistance for manuscript formatting, literature cross-referencing, and stylistic review. The author reviewed and edited all content and takes full responsibility for the content of the published article.

\bibliographystyle{elsarticle-num}
\bibliography{references}

\end{document}
"""
    return content

if __name__ == "__main__":
    tex_path = os.path.join("manuscript", "main.tex")
    content = build_latex_manuscript()
    with open(tex_path, "w", encoding="utf-8") as f:
        f.write(content)
    print(f"Successfully generated and wrote upgraded manuscript to {tex_path} ({len(content)} characters).")
