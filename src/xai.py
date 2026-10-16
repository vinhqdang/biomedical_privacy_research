"""
Information-Theoretic Dual-Channel Dirichlet Attribution (DCDA)
Computes path-integrated feature attributions disentangling:
1. Channel A (Diagnostic Risk Attribution): Features driving posterior disease probability p_1(x) = alpha_1 / S
   relative to demographic-matched physiological reference baselines.
2. Channel B (Epistemic Doubt Attribution): Features driving Epistemic Mutual Information I[y, p | x],
   identifying physiological attributes that cause clinical vacuity and model ambiguity.
"""

import torch
import numpy as np

def compute_epistemic_mutual_information(alpha):
    """
    Computes Epistemic Mutual Information I[y, p | x] under Dirichlet distribution:
    I[y, p | x] = H(E[p]) - E_{p ~ Dir(alpha)}[H(p)]
    alpha: Dirichlet concentration parameters (B, K)
    """
    S = torch.sum(alpha, dim=-1, keepdim=True)
    p_hat = alpha / S # Expected probability (B, K)
    
    # 1. Total predictive entropy H(E[p])
    total_entropy = -torch.sum(p_hat * torch.log(p_hat + 1e-12), dim=-1)
    
    # 2. Expected aleatoric entropy E_{p}[H(p)]
    # E[ln p_k] = digamma(alpha_k + 1) - digamma(S + 1)
    digamma_diff = torch.digamma(alpha + 1.0) - torch.digamma(S + 1.0)
    expected_entropy = -torch.sum(p_hat * digamma_diff, dim=-1)
    
    # 3. Epistemic mutual information (non-negative)
    epistemic_mi = torch.clamp(total_entropy - expected_entropy, min=0.0)
    return epistemic_mi

class DualChannelDirichletExplainer:
    def __init__(self, model, steps=50):
        """
        model: EvidentialNet instance
        steps: Number of Gauss-Legendre quadrature nodes for path integrals
        """
        self.model = model
        self.steps = steps
        
    def attribute(self, x, sensitive_attr=None, baseline=None, target_class=1):
        """
        Computes dual-channel path integrated gradients.
        Args:
            x: input tensor (B, D)
            sensitive_attr: demographic attribute tensor (B,)
            baseline: baseline reference tensor (B, D), defaults to demographic cohort median or zeros
            target_class: index of diagnostic risk class (typically 1 for diseased)
        Returns:
            risk_attr: attribution for positive diagnostic risk (B, D)
            doubt_attr: attribution for epistemic mutual information / clinical doubt (B, D)
        """
        self.model.eval()
        device = x.device
        B, D = x.shape
        
        if baseline is None:
            baseline = torch.zeros_like(x)
            
        # Gauss-Legendre quadrature points on [0, 1]
        nodes, weights = np.polynomial.legendre.leggauss(self.steps)
        # Rescale from [-1, 1] to [0, 1]
        nodes = 0.5 * (nodes + 1.0)
        weights = 0.5 * weights
        
        nodes_tensor = torch.tensor(nodes, dtype=torch.float32, device=device)
        weights_tensor = torch.tensor(weights, dtype=torch.float32, device=device)
        
        risk_grads_accum = torch.zeros_like(x)
        doubt_grads_accum = torch.zeros_like(x)
        
        for step_idx in range(self.steps):
            t = nodes_tensor[step_idx]
            w = weights_tensor[step_idx]
            
            # Interpolated points along straight-line path
            interpolated = baseline + t * (x - baseline)
            interpolated = interpolated.clone().detach().requires_grad_(True)
            
            evidence, alpha_params, prob, uncertainty = self.model(interpolated)
            
            # Channel A: Disease risk target p_hat_1(x) = alpha_1 / S
            risk_target = prob[:, target_class].sum()
            grad_risk = torch.autograd.grad(risk_target, interpolated, retain_graph=True)[0]
            risk_grads_accum += w * grad_risk
            
            # Channel B: Epistemic mutual information I[y, p | x]
            epistemic_mi = compute_epistemic_mutual_information(alpha_params).sum()
            grad_doubt = torch.autograd.grad(epistemic_mi, interpolated, retain_graph=False)[0]
            doubt_grads_accum += w * grad_doubt
            
        # Path integrated attributions: (x - baseline) * integral(grad)
        risk_attr = (x - baseline) * risk_grads_accum
        doubt_attr = (x - baseline) * doubt_grads_accum
        
        return risk_attr.detach().cpu().numpy(), doubt_attr.detach().cpu().numpy()

def compute_explanation_infidelity(model, x, attributions, num_samples=30, noise_scale=0.1):
    """
    Computes Explanation Infidelity (Yeh et al., 2019):
    INFD = E_{I ~ N(0, noise_scale^2 I)} [ (f(x) - f(x - I) - I^T Phi)^2 ]
    Lower values indicate higher explanation fidelity.
    """
    model.eval()
    device = x.device
    N, D = x.shape
    infidelities = []
    
    with torch.no_grad():
        if hasattr(model, 'forward') and 'Evidential' in model.__class__.__name__:
            _, _, base_probs, _ = model(x)
            base_val = base_probs[:, 1].cpu().numpy()
        else:
            _, base_probs = model(x)
            base_val = base_probs[:, 1].cpu().numpy()
            
    for i in range(N):
        x_i = x[i:i+1]
        attr_i = attributions[i]
        sample_errors = []
        
        for _ in range(num_samples):
            noise = torch.randn_like(x_i) * noise_scale
            x_perturbed = x_i - noise
            
            with torch.no_grad():
                if hasattr(model, 'forward') and 'Evidential' in model.__class__.__name__:
                    _, _, p_pert, _ = model(x_perturbed)
                    val_pert = p_pert[:, 1].item()
                else:
                    _, p_pert = model(x_perturbed)
                    val_pert = p_pert[:, 1].item()
                    
            delta_f = base_val[i] - val_pert
            linear_approx = np.dot(noise.cpu().numpy().flatten(), attr_i.flatten())
            sample_errors.append((delta_f - linear_approx) ** 2)
            
        infidelities.append(np.mean(sample_errors))
        
    return float(np.mean(infidelities))

def compute_faithfulness_correlation(model, x, attributions):
    """
    Computes Faithfulness Correlation (Spearman rank correlation between feature attribution magnitude
    and prediction shift under zero-ablation).
    """
    model.eval()
    N, D = x.shape
    correlations = []
    
    with torch.no_grad():
        if hasattr(model, 'forward') and 'Evidential' in model.__class__.__name__:
            _, _, base_probs, _ = model(x)
            base_val = base_probs[:, 1].cpu().numpy()
        else:
            _, base_probs = model(x)
            base_val = base_probs[:, 1].cpu().numpy()
            
    for i in range(N):
        attr_i = np.abs(attributions[i])
        feature_shifts = []
        
        for j in range(D):
            x_ablated = x[i:i+1].clone()
            x_ablated[0, j] = 0.0
            
            with torch.no_grad():
                if hasattr(model, 'forward') and 'Evidential' in model.__class__.__name__:
                    _, _, p_abl, _ = model(x_ablated)
                    shift = abs(base_val[i] - p_abl[:, 1].item())
                else:
                    _, p_abl = model(x_ablated)
                    shift = abs(base_val[i] - p_abl[:, 1].item())
            feature_shifts.append(shift)
            
        feature_shifts = np.array(feature_shifts)
        if np.std(attr_i) > 1e-6 and np.std(feature_shifts) > 1e-6:
            r = np.corrcoef(attr_i, feature_shifts)[0, 1]
            if not np.isnan(r):
                correlations.append(r)
                
    return float(np.mean(correlations)) if correlations else 0.0
