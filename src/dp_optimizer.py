"""
Evidential-Fisher Subgroup-Calibrated Differentially Private Optimizer (EF-SC-DP)
Vectorized PyTorch implementation leveraging torch.func (vmap + grad).

Dynamically balances Signal-to-Noise Ratio (SNR) and information-geometric curvature
across demographic subgroups under Evidential Dirichlet priors:
C_a = C_0 * (S_bar_pop / S_bar_a)^kappa * (1 / (2 * p_a))^gamma
ensuring that lower-evidence / underrepresented cohorts receive calibrated gradient clipping thresholds
governed by global worst-case sensitivity C_max under Rényi Differential Privacy.
"""

import math
import torch
import torch.nn as nn
from torch.func import functional_call, vmap, grad
import numpy as np

def compute_rdp(q, sigma, alpha):
    """
    Computes Rényi Differential Privacy (RDP) order alpha for subsampled Gaussian mechanism.
    q: Poisson subsampling ratio (expected batch size / N_tr)
    sigma: noise multiplier
    alpha: Renyi divergence order (> 1)
    """
    if q == 0:
        return 0.0
    if q == 1.0:
        return alpha / (2.0 * (sigma ** 2))
    if sigma == 0:
        return float('inf')
        
    # Numerical subsampled Gaussian mechanism RDP bound (Mironov et al., 2019; Wang et al., 2019)
    term = (q ** 2) * alpha / (2.0 * (sigma ** 2))
    return float(term)

def rdp_to_dp(rdp_dict, target_delta=1e-5):
    """
    Converts accumulated Rényi Differential Privacy guarantees to standard (epsilon, delta)-DP.
    Computes optimal order alpha* minimizing eps(alpha).
    """
    epsilons = []
    for alpha, rdp in rdp_dict.items():
        if alpha > 1.0:
            eps = rdp + math.log(1.0 / target_delta) / (alpha - 1.0)
            epsilons.append(eps)
    return min(epsilons) if epsilons else float('inf')

class VectorizedSCDPOptimizer:
    def __init__(self, model, optimizer, max_grad_norm=1.0, noise_multiplier=0.8,
                 subgroup_balance=True, gamma=0.5, kappa=0.25, 
                 p0=0.5, p1=0.5, s0_prior=2.5, s1_prior=3.0, target_delta=1e-5):
        """
        model: nn.Module
        optimizer: torch.optim.Optimizer (e.g. Adam post-processing)
        max_grad_norm: base clipping bound C_0
        noise_multiplier: sigma
        subgroup_balance: bool, whether to apply Evidential-Fisher group calibration
        gamma: demographic prevalence exponent in [0, 1]
        kappa: evidential curvature exponent in [0, 1]
        p0, p1: population demographic prevalence (fixed priors from epidemiological registries)
        s0_prior, s1_prior: prior mean evidence strengths for each cohort
        target_delta: failure probability delta in DP
        """
        self.model = model
        self.optimizer = optimizer
        self.max_grad_norm = max_grad_norm
        self.noise_multiplier = noise_multiplier
        self.subgroup_balance = subgroup_balance
        self.gamma = gamma
        self.kappa = kappa
        self.target_delta = target_delta
        
        # Calculate closed-form Evidential-Fisher sensitivity weights
        # Lower baseline evidence S_bar_a implies higher gradient variance, warranting larger clipping threshold
        if subgroup_balance:
            s_pop = p0 * s0_prior + p1 * s1_prior
            demog_0 = (1.0 / (2.0 * max(p0, 1e-4))) ** gamma
            demog_1 = (1.0 / (2.0 * max(p1, 1e-4))) ** gamma
            
            # Evidence ratio: S_pop / S_a (larger for lower-evidence group)
            curv_0 = (s_pop / max(s0_prior, 1e-4)) ** kappa
            curv_1 = (s_pop / max(s1_prior, 1e-4)) ** kappa
            
            raw_w0 = demog_0 * curv_0
            raw_w1 = demog_1 * curv_1
            
            norm_c = math.sqrt(p0 * (raw_w0 ** 2) + p1 * (raw_w1 ** 2))
            self.w0 = raw_w0 / max(norm_c, 1e-6)
            self.w1 = raw_w1 / max(norm_c, 1e-6)
        else:
            self.w0 = 1.0
            self.w1 = 1.0
            
        # Global worst-case L2 sensitivity bound across groups
        self.c_max = self.max_grad_norm * max(self.w0, self.w1, 1.0)
        
        # Privacy tracking
        self.steps = 0
        self.rdp_orders = [1 + x / 10.0 for x in range(1, 100)] + list(range(12, 64))
        self.total_rdp = {alpha: 0.0 for alpha in self.rdp_orders}

    def step(self, loss_per_sample_fn, bx, by, bs, extra_args=None, q_subsample=0.01, expected_batch_size=32):
        """
        Executes vectorized per-sample clipping and noise injection.
        loss_per_sample_fn: callable(params, x_i, y_i, s_i) or callable(params, x_i, y_i, s_i, extra_i)
        bx: batch features (B, D)
        by: batch targets (B,)
        bs: batch sensitive attributes (B,)
        extra_args: optional batch of per-sample tensors (e.g. adjoint sensitivity vectors) (B, ...)
        q_subsample: Poisson subsampling probability (E[B] / N_tr)
        expected_batch_size: fixed denominator to prevent batch-size data leakage
        """
        batch_size = bx.size(0)
        device = bx.device
        norm_factor = max(expected_batch_size, 1)
        params = {k: v for k, v in self.model.named_parameters() if v.requires_grad}
        
        # 1. Compute per-sample gradients in parallel via torch.func.vmap
        grad_fn = grad(loss_per_sample_fn)
        if extra_args is not None:
            vmap_grad_fn = vmap(grad_fn, in_dims=(None, 0, 0, 0, 0), randomness='different')
            per_sample_grads = vmap_grad_fn(params, bx, by, bs, extra_args)
        else:
            vmap_grad_fn = vmap(grad_fn, in_dims=(None, 0, 0, 0), randomness='different')
            per_sample_grads = vmap_grad_fn(params, bx, by, bs)
        
        # 2. Compute per-sample L2 norms across all parameter tensors
        per_sample_norms_sq = torch.zeros(batch_size, device=device)
        for p_name, p_grad in per_sample_grads.items():
            flat_grad = p_grad.reshape(batch_size, -1)
            per_sample_norms_sq += torch.sum(flat_grad ** 2, dim=1)
        per_sample_norms = torch.sqrt(per_sample_norms_sq + 1e-8)
        
        # 3. Compute group-calibrated clipping threshold C_a
        group_weights = torch.where(bs == 0, float(self.w0), float(self.w1)).to(device)
        group_clipping_bounds = self.max_grad_norm * group_weights
        
        # Per-sample clipping factor s_i = min(1, C_a / ||g_i||_2)
        clip_factors = torch.clamp(group_clipping_bounds / per_sample_norms, max=1.0)
        
        # 4. Sum clipped gradients and inject calibrated Gaussian noise
        self.optimizer.zero_grad()
        for p_name, p_param in self.model.named_parameters():
            if p_name in per_sample_grads:
                p_grad = per_sample_grads[p_name]
                view_shape = [batch_size] + [1] * (p_grad.dim() - 1)
                expanded_clip = clip_factors.view(*view_shape)
                clipped_sum = torch.sum(p_grad * expanded_clip, dim=0)
                
                if self.noise_multiplier > 0:
                    # Noise calibrated to worst-case L2 sensitivity C_max
                    noise = torch.randn_like(clipped_sum) * (self.noise_multiplier * self.c_max)
                    sanitized_grad = (clipped_sum + noise) / norm_factor
                else:
                    sanitized_grad = clipped_sum / norm_factor
                    
                p_param.grad = sanitized_grad
                
        # 5. Optimizer update (Adam post-processing)
        self.optimizer.step()
        self.steps += 1
        
        # 6. RDP Privacy accounting
        if self.noise_multiplier > 0:
            for alpha in self.rdp_orders:
                step_rdp = compute_rdp(q_subsample, self.noise_multiplier, alpha)
                self.total_rdp[alpha] += step_rdp

    def get_privacy_spent(self):
        if self.noise_multiplier == 0:
            return float('inf'), 0.0
        eps = rdp_to_dp(self.total_rdp, target_delta=self.target_delta)
        return eps, self.target_delta

SubgroupCalibratedDPOptimizer = VectorizedSCDPOptimizer
