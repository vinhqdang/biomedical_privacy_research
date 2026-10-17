"""
Differentially Private Group-Conditional Conformal Prediction (DP-GCCP)
via Disjoint Histogram Mechanism and PAC Concentration Bounds.

Disjoint-bin histogram mechanism with Laplace noise, cumulative summation, and optional
Dvoretzky-Kiefer-Wolfowitz plus Bernstein PAC buffers (see the class docstring).
"""

import math
import numpy as np

class DPGroupConditionalConformalPredictor:
    """
    Differentially private split-conformal calibration on a public histogram grid.

    Privacy (bounded replacement adjacency, calibration-set sizes n_a treated as public): per-bin counts have
    l1 sensitivity 2, so Laplace(2/eps) noise per bin gives (eps, 0)-DP for one released histogram; the
    threshold is post-processing.

    Modes
      "pac"    : threshold at the level 1 - alpha + tau, with tau the DKW + Laplace-Bernstein buffer of
                 Theorem 3 and F~(t_m) = (sum_{j<=m} c~_j) / n (no clamping, no renormalization).
                 With probability >= 1 - beta the conditional coverage is >= 1 - alpha.
      "plugin" : threshold at the finite-sample level (1 - alpha)(1 + 1/n) on the same noisy CDF, with no buffer.
                 No finite-sample guarantee under the Laplace noise; used as the practical variant.
    Group-conditional calibration (use_group=True) shrinks each group threshold toward a marginal threshold,
    q_a = lam_a q_a^grp + (1 - lam_a) q^marg, lam_a = n_a / (n_a + kappa); the budget is split eps/2 for the
    marginal histogram and eps/2 for the group histograms (disjoint groups, parallel composition).
    """
    def __init__(self, alpha=0.10, epsilon_conf=1.0, beta=0.05, num_bins=20, kappa=15.0, mode="plugin"):
        self.alpha = alpha
        self.epsilon_conf = epsilon_conf
        self.beta = beta
        self.num_bins = num_bins
        self.kappa = kappa
        self.mode = mode
        self.bin_edges = np.linspace(0.0, 1.0, num_bins + 1)
        self.grid = self.bin_edges[1:]
        self.group_quantiles = {}
        self.marginal_quantile = 1.0
        self.group_sizes = {}

    def calibrate(self, probs, y_true, sensitive_attrs, private=True, use_group=True):
        N = probs.shape[0]
        scores = np.clip(1.0 - probs[np.arange(N), y_true], 0.0, 1.0)
        eps_marg = self.epsilon_conf / 2.0 if (use_group and private) else self.epsilon_conf
        self.marginal_quantile = self._quantile(scores, eps_marg, private)
        self.group_quantiles = {}
        for g in np.unique(sensitive_attrs):
            gs = scores[sensitive_attrs == g]
            self.group_sizes[int(g)] = len(gs)
            if not use_group:
                self.group_quantiles[int(g)] = self.marginal_quantile
                continue
            q_g = self._quantile(gs, self.epsilon_conf / 2.0 if private else None, private)
            lam = len(gs) / (len(gs) + self.kappa)
            self.group_quantiles[int(g)] = float(np.clip(lam * q_g + (1.0 - lam) * self.marginal_quantile, 0.0, 1.0))

    def _quantile(self, scores, epsilon, private):
        n = len(scores)
        if n == 0:
            return 1.0
        M = self.num_bins
        counts, _ = np.histogram(scores, bins=self.bin_edges)
        counts = counts.astype(np.float64)
        if not private:
            # exact split-conformal quantile (no histogram grid) for non-private baselines
            level = min(1.0, (1.0 - self.alpha) * (1.0 + 1.0 / n))
            return float(np.quantile(scores, level, method="higher"))
        noisy = counts + np.random.laplace(0.0, 2.0 / epsilon, size=M)
        cdf = np.cumsum(noisy) / n
        if self.mode == "pac":
            beta = max(self.beta, 1e-6)
            tau = (math.sqrt(math.log(4.0 / beta) / (2.0 * n))
                   + (2.0 * 2.0 * math.sqrt(M * math.log(4.0 * M / beta)) + 2.0 * 2.0 * math.log(4.0 * M / beta)) / (n * epsilon))
            level = 1.0 - self.alpha + tau
        else:
            level = (1.0 - self.alpha) * (1.0 + 1.0 / n)
        idx = np.where(cdf >= level)[0]
        return float(self.grid[idx[0]]) if len(idx) else 1.0

    def predict(self, probs, sensitive_attrs=None, use_group_conditional=True):
        """
        Constructs prediction sets C(X) = {k : 1 - p_k <= q_thresh}.
        """
        N, K = probs.shape
        prediction_sets = []
        set_sizes = np.zeros(N, dtype=int)
        
        for i in range(N):
            g = sensitive_attrs[i] if (sensitive_attrs is not None and use_group_conditional) else None
            q_thresh = self.group_quantiles.get(g, self.marginal_quantile) if g is not None else self.marginal_quantile
            
            cand_classes = [k for k in range(K) if (1.0 - probs[i, k]) <= q_thresh]
            if len(cand_classes) == 0:
                cand_classes = [int(np.argmax(probs[i]))]
                
            prediction_sets.append(cand_classes)
            set_sizes[i] = len(cand_classes)
            
        return prediction_sets, set_sizes

    def evaluate_coverage(self, prediction_sets, y_true, sensitive_attrs=None):
        N = len(y_true)
        covered = np.array([y_true[i] in prediction_sets[i] for i in range(N)], dtype=float)
        marginal_coverage = float(np.mean(covered))
        
        set_sizes = np.array([len(s) for s in prediction_sets], dtype=float)
        avg_set_size = float(np.mean(set_sizes))
        singlet_rate = float(np.mean(set_sizes == 1.0))
        
        group_coverages = {}
        max_disparity = 0.0
        
        if sensitive_attrs is not None:
            unique_groups = np.unique(sensitive_attrs)
            covs = []
            for g in unique_groups:
                mask = (sensitive_attrs == g)
                if np.sum(mask) > 0:
                    cov_g = float(np.mean(covered[mask]))
                    group_coverages[int(g)] = cov_g
                    covs.append(cov_g)
            if len(covs) >= 2:
                max_disparity = float(max(covs) - min(covs))
                
        return {
            "marginal_coverage": marginal_coverage,
            "avg_set_size": avg_set_size,
            "singlet_rate": singlet_rate,
            "group_coverages": group_coverages,
            "max_coverage_disparity": max_disparity
        }
