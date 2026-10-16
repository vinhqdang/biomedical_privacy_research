"""
Class-Conditional Latent Sliced Wasserstein Transport (CC-SWD)
Alters latent clinical feature geometry to enforce Equalized Odds (separation)
rather than Demographic Parity (independence).
Aligns latent feature representations across demographic subgroups *conditioned on true clinical outcome*,
preserving true pathophysiological diagnostic signals while eliminating spurious demographic bias.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

def class_conditional_sliced_wasserstein_distance(latent, y, sensitive_attr, num_projections=50, p=1):
    """
    Computes Class-Conditional Sliced Wasserstein Distance (CC-SWD).
    Separately aligns P(h(X) | A=0, Y=y) with P(h(X) | A=1, Y=y) for each diagnostic class y in {0, 1}.
    This directly enforces Equalized Odds sufficiency/separation.
    Args:
        latent: Tensor of latent representations (B, D)
        y: Ground truth labels (B,)
        sensitive_attr: Demographic attribute (B,)
        num_projections: Number of random 1D projections on unit sphere
        p: Order of Wasserstein metric (1 for W1)
    """
    device = latent.device
    d = latent.size(1)
    unique_classes = torch.unique(y)
    
    total_loss = torch.tensor(0.0, device=device, requires_grad=True)
    valid_classes = 0
    
    # Generate random projection vectors on unit hypersphere
    projections = torch.randn(d, num_projections, device=device)
    projections = projections / torch.norm(projections, dim=0, keepdim=True)
    
    for c in unique_classes:
        mask_0_c = (sensitive_attr == 0) & (y == c)
        mask_1_c = (sensitive_attr == 1) & (y == c)
        
        n0 = mask_0_c.sum().item()
        n1 = mask_1_c.sum().item()
        
        if n0 < 2 or n1 < 2:
            continue
            
        z0 = latent[mask_0_c]
        z1 = latent[mask_1_c]
        
        # Project representations to 1D
        proj_0 = torch.matmul(z0, projections) # (n0, num_projections)
        proj_1 = torch.matmul(z1, projections) # (n1, num_projections)
        
        # Sort 1D projections
        sorted_0, _ = torch.sort(proj_0, dim=0)
        sorted_1, _ = torch.sort(proj_1, dim=0)
        
        # Match quantile grid
        quantiles = torch.linspace(0.0, 1.0, steps=min(n0, n1), device=device)
        idx_0 = (quantiles * (n0 - 1)).long()
        idx_1 = (quantiles * (n1 - 1)).long()
        
        q_0 = sorted_0[idx_0]
        q_1 = sorted_1[idx_1]
        
        if p == 1:
            class_swd = torch.mean(torch.abs(q_0 - q_1))
        else:
            class_swd = torch.mean((q_0 - q_1) ** 2)
            
        total_loss = total_loss + class_swd
        valid_classes += 1
        
    return total_loss / max(valid_classes, 1)

def sliced_wasserstein_distance(x_0, x_1, num_projections=50):
    """
    Standard marginal Sliced Wasserstein Distance between two sample distributions.
    Maintained for baseline comparison.
    """
    if x_0.size(0) < 2 or x_1.size(0) < 2:
        return torch.tensor(0.0, device=x_0.device, requires_grad=True)
        
    d = x_0.size(1)
    projections = torch.randn(d, num_projections, device=x_0.device)
    projections = projections / torch.norm(projections, dim=0, keepdim=True)
    
    proj_0 = torch.matmul(x_0, projections)
    proj_1 = torch.matmul(x_1, projections)
    
    sorted_0, _ = torch.sort(proj_0, dim=0)
    sorted_1, _ = torch.sort(proj_1, dim=0)
    
    n0, n1 = x_0.size(0), x_1.size(0)
    quantiles = torch.linspace(0.0, 1.0, steps=min(n0, n1), device=x_0.device)
    idx_0 = (quantiles * (n0 - 1)).long()
    idx_1 = (quantiles * (n1 - 1)).long()
    
    swd = torch.mean(torch.abs(sorted_0[idx_0] - sorted_1[idx_1]))
    return swd

def equalized_odds_penalty(prob, y, sensitive_attr):
    """
    Differentiable surrogate penalty for Equalized Odds disparity.
    Penalizes disparity in predicted positive rates conditioned on true label y:
    | E[P(Y=1|A=0, Y=y)] - E[P(Y=1|A=1, Y=y)] |^2 for y in {0, 1}.
    """
    device = prob.device
    penalty = torch.tensor(0.0, device=device, requires_grad=True)
    p1 = prob[:, 1] if prob.dim() > 1 else prob
    
    for c in [0, 1]:
        m0 = (sensitive_attr == 0) & (y == c)
        m1 = (sensitive_attr == 1) & (y == c)
        if m0.sum() > 0 and m1.sum() > 0:
            diff = torch.mean(p1[m0]) - torch.mean(p1[m1])
            penalty = penalty + (diff ** 2)
            
    return penalty

def demographic_parity_loss(prob, sensitive_attr):
    """
    Penalizes difference in mean predicted positive probability across demographic groups.
    """
    mask_0 = (sensitive_attr == 0)
    mask_1 = (sensitive_attr == 1)
    if mask_0.sum() == 0 or mask_1.sum() == 0:
        return torch.tensor(0.0, device=prob.device, requires_grad=True)
        
    p1 = prob[:, 1] if prob.dim() > 1 else prob
    mean_0 = torch.mean(p1[mask_0])
    mean_1 = torch.mean(p1[mask_1])
    return (mean_0 - mean_1) ** 2

def compute_latent_cc_swd_adjoints(model, bx, by, bs, num_projections=50):
    """
    Computes adjoint sensitivity vectors u_i = d(L_CC-SWD)/d(h_i)
    for exact per-sample gradient decomposition under DP-SGD.
    Enables true batch-level Sliced Wasserstein Distance alignment on latent representations
    to be trained within vectorized per-sample clipping DP-SGD without auxiliary privacy loss.
    """
    h = model.forward_latent(bx)
    if h.requires_grad:
        h_target = h
    else:
        h_target = h.clone().detach().requires_grad_(True)
        
    swd_loss = class_conditional_sliced_wasserstein_distance(
        h_target, by, bs, num_projections=num_projections
    )
    if swd_loss.requires_grad and swd_loss.item() > 0:
        adjoints = torch.autograd.grad(swd_loss, h_target, retain_graph=False)[0].detach()
    else:
        adjoints = torch.zeros_like(h)
    return adjoints

class PrivatizedClassConditionalAnchorAligner(nn.Module):
    """
    Privatized Class-Conditional Prototypical Anchor Transport (PC-PAT).
    Provides strict per-sample record independence for Theorem 1 in DP-SGD.
    Maintains invariant diagnostic class prototypes mu_y for each outcome class y in {0, 1}
    as learnable network parameters (or data-independent orthogonal prior anchors).
    
    Each patient representation h(x_i) is aligned toward the target class prototype:
    ell_PAT(x_i, y_i) = 0.5 * ||h(x_i) - mu_{y_i}||_2^2
    
    Because mu_y are model parameters updated jointly via privatized gradient descent,
    the per-sample gradient depends strictly and exclusively on sample i alone:
    g_i = grad_{(theta, mu)} [ L_EDL(x_i, y_i) + lambda_fair * ell_PAT(x_i, y_i) ]
    guaranteeing that the L2 sensitivity under add/remove adjacency is unconditionally bounded by C_max.
    """
    def __init__(self, latent_dim=32, num_classes=2, init_orthogonal=True):
        super(PrivatizedClassConditionalAnchorAligner, self).__init__()
        self.latent_dim = latent_dim
        self.num_classes = num_classes
        
        # Initialize prototypes with orthogonal unit vectors or standard normal
        if init_orthogonal and latent_dim >= num_classes:
            init_weights = torch.zeros(num_classes, latent_dim)
            for c in range(num_classes):
                init_weights[c, c] = 1.0
        else:
            init_weights = torch.randn(num_classes, latent_dim) * 0.1
            
        self.prototypes = nn.Parameter(init_weights)

    def per_sample_anchor_loss(self, h_i, y_i):
        """
        Computes per-sample anchor distance. Depends strictly on sample i alone.
        h_i: (latent_dim,) or (1, latent_dim)
        y_i: scalar or (1,) class index
        """
        c = int(y_i)
        proto = self.prototypes[c]
        diff = h_i.squeeze() - proto
        return 0.5 * torch.sum(diff ** 2)

    def forward(self, h, y):
        """
        Batch-level forward pass for non-private baselines.
        h: (B, latent_dim)
        y: (B,)
        """
        target_protos = self.prototypes[y] # (B, latent_dim)
        return 0.5 * torch.mean(torch.sum((h - target_protos) ** 2, dim=1))



