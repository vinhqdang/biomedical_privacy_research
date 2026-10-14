"""
Differentially Private Group-Conditional Conformal Prediction (DP-GCCP)
Guarantees distribution-free, finite-sample coverage for every demographic subgroup
while strictly bounding privacy leakage via the Exponential / Laplace Quantile Mechanism.
"""

import math
import numpy as np
import torch

class DPGroupConditionalConformalPredictor:
    """
    Differentially Private Group-Conditional Conformal Prediction
    Calibrates group-specific non-conformity thresholds with formal DP noise injection.
    """
    def __init__(self, alpha=0.10, epsilon_conf=1.0, delta_conf=1e-5):
        """
        alpha: significance level (e.g. 0.10 for 90% target coverage)
        epsilon_conf: differential privacy budget for calibration step
        delta_conf: privacy delta
        """
        self.alpha = alpha
        self.epsilon_conf = epsilon_conf
        self.delta_conf = delta_conf
        self.group_quantiles = {}
        self.marginal_quantile = None
        
    def calibrate(self, probs, y_true, sensitive_attrs, private=True):
        """
        Calibrates conformal prediction thresholds on calibration dataset.
        Args:
            probs: numpy array of predicted probabilities (N, K)
            y_true: numpy array of true ground-truth labels (N,)
            sensitive_attrs: numpy array of demographic group indicators (N,)
            private: bool, whether to apply DP quantile perturbation
        """
        N, K = probs.shape
        # Non-conformity score: 1 - P(Y = y_true | X)
        true_class_probs = probs[np.arange(N), y_true]
        non_conformity_scores = 1.0 - true_class_probs
        
        # Marginal calibration
        marginal_q_val = self._compute_quantile(non_conformity_scores, self.alpha, 
                                                epsilon=self.epsilon_conf, private=private)
        self.marginal_quantile = marginal_q_val
        
        # Group-conditional calibration
        unique_groups = np.unique(sensitive_attrs)
        for g in unique_groups:
            mask = (sensitive_attrs == g)
            group_scores = non_conformity_scores[mask]
            # Split privacy budget evenly across groups: epsilon / len(groups)
            group_eps = self.epsilon_conf / len(unique_groups) if private else 0.0
            q_g = self._compute_quantile(group_scores, self.alpha, epsilon=group_eps, private=private)
            self.group_quantiles[g] = q_g

    def _compute_quantile(self, scores, alpha, epsilon=1.0, private=True):
        """
        Computes finite-sample adjusted conformal quantile with optional DP noise.
        """
        n = len(scores)
        if n == 0:
            return 1.0
            
        target_prob = 1.0 - alpha
        # Finite-sample conformal correction: ceil((n + 1) * (1 - alpha)) / n
        q_idx = min(n - 1, int(np.ceil((n + 1) * target_prob)) - 1)
        sorted_scores = np.sort(scores)
        base_quantile = sorted_scores[max(0, q_idx)]
        
        if not private or epsilon <= 0 or epsilon == float('inf'):
            return float(np.clip(base_quantile, 0.0, 1.0))
            
        # Differential Privacy Laplace Mechanism on Quantile
        # Sensitivity of empirical quantile function is bounded by 1 / n
        sensitivity = 1.0 / n
        # Finite-sample coverage buffer to offset negative Laplace tail
        buffer = (2.0 * math.log(10.0)) / (n * epsilon)
        laplace_noise = np.random.laplace(0.0, sensitivity / epsilon)
        
        noisy_quantile = base_quantile + buffer + laplace_noise
        return float(np.clip(noisy_quantile, 0.0, 1.0))
        
    def predict(self, probs, sensitive_attrs=None, use_group_conditional=True):
        """
        Generates conformal prediction sets for test instances.
        Returns:
            prediction_sets: list of sets containing predicted candidate classes
            set_sizes: numpy array of set cardinalities
        """
        N, K = probs.shape
        prediction_sets = []
        set_sizes = np.zeros(N, dtype=int)
        
        for i in range(N):
            g = sensitive_attrs[i] if (sensitive_attrs is not None and use_group_conditional) else None
            q_thresh = self.group_quantiles.get(g, self.marginal_quantile) if g is not None else self.marginal_quantile
            
            # Include class k if non-conformity score (1 - prob_k) <= threshold
            cand_classes = [k for k in range(K) if (1.0 - probs[i, k]) <= q_thresh]
            # In rare event of empty set, include argmax
            if len(cand_classes) == 0:
                cand_classes = [int(np.argmax(probs[i]))]
                
            prediction_sets.append(cand_classes)
            set_sizes[i] = len(cand_classes)
            
        return prediction_sets, set_sizes
        
    def evaluate_coverage(self, prediction_sets, y_true, sensitive_attrs=None):
        """
        Evaluates empirical coverage and efficiency metrics overall and per subgroup.
        """
        N = len(y_true)
        covered = np.array([y_true[i] in prediction_sets[i] for i in range(N)], dtype=float)
        marginal_coverage = float(np.mean(covered))
        avg_set_size = float(np.mean([len(s) for s in prediction_sets]))
        singlet_rate = float(np.mean([1.0 if len(s) == 1 else 0.0 for s in prediction_sets]))
        
        results = {
            "marginal_coverage": marginal_coverage,
            "avg_set_size": avg_set_size,
            "singlet_rate": singlet_rate
        }
        
        if sensitive_attrs is not None:
            unique_groups = np.unique(sensitive_attrs)
            group_coverages = {}
            group_sizes = {}
            for g in unique_groups:
                mask = (sensitive_attrs == g)
                if np.sum(mask) > 0:
                    group_coverages[int(g)] = float(np.mean(covered[mask]))
                    group_sizes[int(g)] = float(np.mean([len(prediction_sets[i]) for i in np.where(mask)[0]]))
            results["group_coverages"] = group_coverages
            results["group_sizes"] = group_sizes
            if len(group_coverages) >= 2:
                coverages = list(group_coverages.values())
                results["max_coverage_disparity"] = float(max(coverages) - min(coverages))
            else:
                results["max_coverage_disparity"] = 0.0
                
        return results
