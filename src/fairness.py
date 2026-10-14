"""
Algorithmic Fairness and Latent Representation Alignment Module
Implements Sliced Wasserstein Distance (SWD) and Group Disparity Penalties.
SWD aligns the distribution of latent representations across demographic subgroups
without the adversarial minimax instability of GAN-based debiasing.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

def sliced_wasserstein_distance(x_0, x_1, num_projections=50):
    """
    Computes 1D Sliced Wasserstein Distance between two sample distributions x_0 and x_1.
    Args:
        x_0: Tensor of shape (N0, D) for subgroup 0
        x_1: Tensor of shape (N1, D) for subgroup 1
        num_projections: Number of random 1D projections
    Returns:
        Scalar SWD loss
    """
    if x_0.size(0) < 2 or x_1.size(0) < 2:
        return torch.tensor(0.0, device=x_0.device, requires_grad=True)
        
    d = x_0.size(1)
    # Generate random projection vectors on unit sphere
    projections = torch.randn(d, num_projections, device=x_0.device)
    projections = projections / torch.norm(projections, dim=0, keepdim=True)
    
    # Project representations to 1D
    proj_0 = torch.matmul(x_0, projections) # (N0, num_projections)
    proj_1 = torch.matmul(x_1, projections) # (N1, num_projections)
    
    # Sort 1D projections
    sorted_0, _ = torch.sort(proj_0, dim=0)
    sorted_1, _ = torch.sort(proj_1, dim=0)
    
    # Interpolate to matching quantile grid if batch sizes differ
    n0, n1 = x_0.size(0), x_1.size(0)
    if n0 != n1:
        quantiles = torch.linspace(0.0, 1.0, steps=min(n0, n1), device=x_0.device)
        idx_0 = (quantiles * (n0 - 1)).long()
        idx_1 = (quantiles * (n1 - 1)).long()
        sorted_0 = sorted_0[idx_0]
        sorted_1 = sorted_1[idx_1]
        
    # L1 Wasserstein distance across 1D slices
    swd = torch.mean(torch.abs(sorted_0 - sorted_1))
    return swd

def demographic_parity_loss(prob, sensitive_attr):
    """
    Penalizes difference in mean predicted positive probability across demographic groups:
    | E[P(Y=1|A=0)] - E[P(Y=1|A=1)] |^2
    """
    mask_0 = (sensitive_attr == 0)
    mask_1 = (sensitive_attr == 1)
    
    if mask_0.sum() == 0 or mask_1.sum() == 0:
        return torch.tensor(0.0, device=prob.device, requires_grad=True)
        
    p1 = prob[:, 1] if prob.dim() > 1 else prob
    mean_0 = torch.mean(p1[mask_0])
    mean_1 = torch.mean(p1[mask_1])
    return (mean_0 - mean_1) ** 2
