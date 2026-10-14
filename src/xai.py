"""
Explainable AI (XAI) Module: Dual-Channel Dirichlet Attribution (DCDA)
Computes path-integrated feature attributions disentangling:
1. Diagnostic Evidence Attribution (features driving the positive disease prediction)
2. Epistemic Uncertainty Attribution (features driving clinical doubt/vacuity)
Includes quantitative evaluation metrics: Explanation Infidelity and Faithfulness Correlation.
"""

import torch
import numpy as np

class DualChannelDirichletExplainer:
    def __init__(self, model, steps=50):
        """
        model: EvidentialNet instance
        steps: Riemann integration steps for path integrals
        """
        self.model = model
        self.steps = steps
        
    def attribute(self, x, baseline=None, target_class=1):
        """
        Computes dual-channel path integrated gradients.
        Args:
            x: input tensor (1, D) or (B, D)
            baseline: baseline reference tensor (same shape), defaults to zeros
            target_class: index of diagnostic target class
        Returns:
            diag_attr: attribution for positive diagnostic evidence (B, D)
            unc_attr: attribution for epistemic uncertainty (B, D)
        """
        self.model.eval()
        if baseline is None:
            baseline = torch.zeros_like(x)
            
        alphas = torch.linspace(0.0, 1.0, self.steps, device=x.device)
        
        diag_grads = []
        unc_grads = []
        
        for alpha in alphas:
            interpolated = baseline + alpha * (x - baseline)
            interpolated = interpolated.clone().detach().requires_grad_(True)
            
            evidence, alpha_params, prob, uncertainty = self.model(interpolated)
            
            # 1. Diagnostic evidence gradient: log(alpha_target)
            target_evidence = torch.log(alpha_params[:, target_class] + 1e-6)
            grad_diag = torch.autograd.grad(target_evidence.sum(), interpolated, retain_graph=True)[0]
            diag_grads.append(grad_diag)
            
            # 2. Epistemic uncertainty gradient: u(x)
            grad_unc = torch.autograd.grad(uncertainty.sum(), interpolated, retain_graph=False)[0]
            unc_grads.append(grad_unc)
            
        # Riemann sum approximation
        avg_diag_grad = torch.stack(diag_grads, dim=0).mean(dim=0)
        avg_unc_grad = torch.stack(unc_grads, dim=0).mean(dim=0)
        
        diag_attr = (x - baseline) * avg_diag_grad
        unc_attr = (x - baseline) * avg_unc_grad
        
        return diag_attr.detach().cpu().numpy(), unc_attr.detach().cpu().numpy()

def compute_explanation_infidelity(model, x, attributions, num_samples=30, noise_scale=0.1):
    """
    Computes Explanation Infidelity:
    INFD = E_{I ~ N(0, noise_scale^2 I)} [ (f(x) - f(x - I) - I^T Phi)^2 ]
    Lower value indicates higher fidelity / faithfulness.
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

def compute_faithfulness_correlation(model, x, attributions, top_k=3):
    """
    Computes Faithfulness Correlation:
    Measures the Spearman rank correlation between the attributed importance of features
    and the actual prediction shift observed when those features are ablated.
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
            x_ablated[0, j] = 0.0 # Zero ablation
            
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
