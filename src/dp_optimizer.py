"""
Vectorized differentially private optimizers (torch.func vmap + grad) with RDP accounting.

Modes
  uniform : standard DP-SGD with one clipping bound C_0.
  refdp   : REF-DP. Group clipping bounds C_a = C_0 * w_a with weights from the Dirichlet Fisher metric,
            w_a = (F_a / F_pop)^(1/2) (1 / (2 p_a))^(1/2), normalized so that sum_a p_a w_a^2 = 1.
            F_a = Tr G(alpha_bar_a) and p_a are estimated from the training data by a Gaussian-mechanism
            release (clipped per-record counts and evidence sums) whose RDP cost is added to the accountant.
  dpsgdf  : DPSGD-F (Xu, Du, Wu), Algorithm 2: per-group bounds from a noisy count of clipped samples,
            C^k = C_0 (1 + (m_k / b_k) / (m / b)), noise scales sigma_1 = 10 sigma_2.
Noise is always calibrated to the worst-case bound max_a C_a. Poisson sampling is done by the caller.
"""

import math
import torch
import torch.nn as nn
from torch.func import functional_call, vmap, grad
import numpy as np

def compute_rdp(q, sigma, alpha):
    """
    RDP of order alpha for the Poisson-subsampled Gaussian mechanism (Mironov et al., 2019),
    computed exactly for integer alpha:
        eps(alpha) = log( sum_k C(alpha,k) (1-q)^(alpha-k) q^k exp((k^2-k)/(2 sigma^2)) ) / (alpha-1)
    q: Poisson sampling probability, sigma: noise multiplier, alpha: integer order >= 2
    """
    if q == 0:
        return 0.0
    if sigma == 0:
        return float('inf')
    if q == 1.0:
        return alpha / (2.0 * (sigma ** 2))
    alpha = int(alpha)
    log_terms = []
    for k in range(alpha + 1):
        log_binom = math.lgamma(alpha + 1) - math.lgamma(k + 1) - math.lgamma(alpha - k + 1)
        log_terms.append(log_binom + (alpha - k) * math.log1p(-q) + (k * math.log(q) if k > 0 else 0.0)
                         + (k * k - k) / (2.0 * sigma ** 2))
    m = max(log_terms)
    log_a = m + math.log(sum(math.exp(t - m) for t in log_terms))
    return log_a / (alpha - 1)

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


def _psi1(x):
    return torch.special.polygamma(1, x)


class VectorizedSCDPOptimizer:
    def __init__(self, model, optimizer, max_grad_norm=1.0, noise_multiplier=0.8, mode="uniform",
                 gamma=0.5, kappa=0.5, alpha_cap=11.0, stats_sigma=3.0, dpsgdf_sigma1_mult=10.0,
                 dpsgdf_ratio_cap=10.0, target_delta=1e-5):
        self.model = model
        self.optimizer = optimizer
        self.max_grad_norm = max_grad_norm
        self.noise_multiplier = noise_multiplier
        self.mode = mode
        self.gamma = gamma
        self.kappa = kappa
        self.alpha_cap = alpha_cap
        self.stats_sigma = stats_sigma
        self.sigma1 = dpsgdf_sigma1_mult * noise_multiplier
        self.ratio_cap = dpsgdf_ratio_cap
        self.target_delta = target_delta

        self.w = [1.0, 1.0]
        self.c_max = self.max_grad_norm
        self.steps = 0
        self.rdp_orders = list(range(2, 129))
        self.total_rdp = {a: 0.0 for a in self.rdp_orders}
        self.last_group_stats = None

    def _add_rdp(self, q, sigma, times=1):
        if sigma <= 0:
            return
        for a in self.rdp_orders:
            self.total_rdp[a] += times * compute_rdp(q, sigma, a)

    def refresh_group_weights(self, X, s):
        """REF-DP: private release of group counts and mean Dirichlet parameters, then Fisher-based weights."""
        if self.mode != "refdp":
            return
        self.model.eval()
        with torch.no_grad():
            _, alpha, _, _ = self.model(X)
        self.model.train()
        e = torch.clamp(alpha - 1.0, 0.0, self.alpha_cap - 1.0)
        K = e.size(1)
        sens = math.sqrt(1.0 + K * (self.alpha_cap - 1.0) ** 2)
        n = torch.zeros(2); esum = torch.zeros(2, K)
        for g in (0, 1):
            m = (s == g)
            n[g] = m.sum().float(); esum[g] = e[m].sum(0)
        noise_sd = self.stats_sigma * sens
        n_t = n + torch.randn(2) * noise_sd
        e_t = esum + torch.randn(2, K) * noise_sd
        self._add_rdp(1.0, self.stats_sigma)
        n_t = torch.clamp(n_t, min=2.0)
        p = n_t / n_t.sum()
        p = torch.clamp(p, 0.02, 0.98); p = p / p.sum()
        a_bar = 1.0 + torch.clamp(e_t / n_t.unsqueeze(1), 0.0, self.alpha_cap - 1.0)
        S_bar = a_bar.sum(1)
        F = _psi1(a_bar).sum(1) - _psi1(S_bar)
        F_pop = float((p * F).sum())
        raw = [((float(F[g]) / max(F_pop, 1e-8)) ** self.kappa) * ((1.0 / (2.0 * float(p[g]))) ** self.gamma) for g in (0, 1)]
        norm = math.sqrt(sum(float(p[g]) * raw[g] ** 2 for g in (0, 1)))
        self.w = [r / max(norm, 1e-8) for r in raw]
        self.c_max = self.max_grad_norm * max(self.w)
        self.last_group_stats = {"p": p.tolist(), "F": F.tolist(), "w": self.w}

    def step(self, loss_per_sample_fn, bx, by, bs, extra_args=None, q_subsample=0.01, expected_batch_size=32,
             extra_grads=None):
        batch_size = bx.size(0)
        device = bx.device
        norm_factor = max(expected_batch_size, 1)
        params = {k: v for k, v in self.model.named_parameters() if v.requires_grad}

        grad_fn = grad(loss_per_sample_fn)
        if extra_args is not None:
            per_sample_grads = vmap(grad_fn, in_dims=(None, 0, 0, 0, 0), randomness='different')(params, bx, by, bs, extra_args)
        else:
            per_sample_grads = vmap(grad_fn, in_dims=(None, 0, 0, 0), randomness='different')(params, bx, by, bs)

        norms_sq = torch.zeros(batch_size, device=device)
        for g_ in per_sample_grads.values():
            norms_sq += torch.sum(g_.reshape(batch_size, -1) ** 2, dim=1)
        norms = torch.sqrt(norms_sq + 1e-8)

        if self.mode == "uniform":
            bounds = torch.full_like(norms, self.max_grad_norm)
            c_max = self.max_grad_norm
        elif self.mode == "refdp":
            wt = torch.tensor(self.w, device=device, dtype=norms.dtype)
            bounds = self.max_grad_norm * wt[bs.long()]
            c_max = self.c_max
        elif self.mode == "dpsgdf":
            C0 = self.max_grad_norm
            over = (norms > C0).float()
            m_k = torch.stack([(over * (bs == g)).sum() for g in (0, 1)])
            o_k = torch.stack([((1 - over) * (bs == g)).sum() for g in (0, 1)])
            m_t = m_k + torch.randn(2, device=device) * self.sigma1
            o_t = o_k + torch.randn(2, device=device) * self.sigma1
            b_t = torch.clamp(m_t + o_t, min=1.0)
            m_tot = torch.clamp(m_t.sum(), min=0.0)
            frac_k = torch.clamp(m_t, min=0.0) / b_t
            frac_pop = torch.clamp(m_tot / float(norm_factor), min=1e-3)
            ratio = torch.clamp(frac_k / frac_pop, 0.0, self.ratio_cap)
            ck = C0 * (1.0 + ratio)
            bounds = ck[bs.long()]
            c_max = float(ck.max())
            self._add_rdp(q_subsample, self.sigma1)
        else:
            raise ValueError(self.mode)

        clip = torch.clamp(bounds / norms, max=1.0)
        self.optimizer.zero_grad()
        for name, p_param in self.model.named_parameters():
            if name in per_sample_grads:
                pg = per_sample_grads[name]
                clipped_sum = torch.sum(pg * clip.view([batch_size] + [1] * (pg.dim() - 1)), dim=0)
                if self.noise_multiplier > 0:
                    clipped_sum = clipped_sum + torch.randn_like(clipped_sum) * (self.noise_multiplier * c_max)
                g_final = clipped_sum / norm_factor
                if extra_grads is not None and name in extra_grads:
                    g_final = g_final + extra_grads[name]
                p_param.grad = g_final
        self.optimizer.step()
        self.steps += 1
        if self.noise_multiplier > 0:
            self._add_rdp(q_subsample, self.noise_multiplier)

    def add_constraint_release(self, q, sigma, times=1):
        """Account an additional Gaussian-mechanism release (e.g., PF-LD constraint gradient or dual update)."""
        self._add_rdp(q, sigma, times)

    def get_privacy_spent(self):
        if self.noise_multiplier == 0:
            return float('inf'), 0.0
        return rdp_to_dp(self.total_rdp, target_delta=self.target_delta), self.target_delta

SubgroupCalibratedDPOptimizer = VectorizedSCDPOptimizer


def accounted_epsilon(q, steps, sigma, streams=1, epoch_releases=0, delta=1e-5):
    """epsilon of `steps` sampled-Gaussian steps (`streams` per step) plus `epoch_releases` full-data Gaussian releases."""
    orders = list(range(2, 129))
    rdp = {a: steps * streams * compute_rdp(q, sigma, a) + epoch_releases * compute_rdp(1.0, sigma, a) for a in orders}
    return rdp_to_dp(rdp, target_delta=delta)


def match_noise_multiplier(target_eps, q, steps, streams, epoch_releases, delta=1e-5):
    """Smallest noise multiplier whose accounted epsilon is at most target_eps (bisection)."""
    lo, hi = 0.3, 50.0
    for _ in range(60):
        mid = 0.5 * (lo + hi)
        if accounted_epsilon(q, steps, mid, streams, epoch_releases, delta) > target_eps:
            lo = mid
        else:
            hi = mid
    return hi
