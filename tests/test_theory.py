"""Numerical checks of the statements made in the manuscript. Run: python tests/test_theory.py"""
import os, sys, math
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
import numpy as np, torch
from scipy.special import polygamma
from src.edl_model import EvidentialNet, edl_bayes_loss
from src.dp_optimizer import compute_rdp, rdp_to_dp
from src.conformal import DPGroupConditionalConformalPredictor
from src.xai import DualChannelDirichletExplainer, compute_epistemic_mutual_information

rng = np.random.default_rng(0)
results = {}

def check(name, ok, detail=""):
    results[name] = ok
    print(("PASS " if ok else "FAIL ") + name + (" | " + detail if detail else ""))

# 1. Dirichlet Fisher metric and its Sherman-Morrison inverse
for K in (2, 3, 5):
    alpha = rng.uniform(1.0, 8.0, K)
    S = alpha.sum()
    G = np.diag(polygamma(1, alpha)) - polygamma(1, S) * np.ones((K, K))
    Ginv_formula = np.diag(1 / polygamma(1, alpha)) + (polygamma(1, S) / (1 - polygamma(1, S) * np.sum(1 / polygamma(1, alpha)))) * np.outer(1 / polygamma(1, alpha), 1 / polygamma(1, alpha))
    err = np.abs(np.linalg.inv(G) - Ginv_formula).max()
    check(f"Sherman-Morrison inverse, K={K}", err < 1e-9, f"max err {err:.2e}")
# Monte Carlo check of G = E[score score^T]
alpha = np.array([2.0, 3.5]); S = alpha.sum()
x = rng.dirichlet(alpha, 400000)
score = np.log(x) - (polygamma(0, alpha) - polygamma(0, S))  # d ln p / d alpha_k
G_mc = score.T @ score / len(x)
G = np.diag(polygamma(1, alpha)) - polygamma(1, S)
check("Fisher metric formula vs Monte Carlo", np.abs(G_mc - G).max() < 0.02, f"max err {np.abs(G_mc-G).max():.3f}")

# 2. Gradient of the Bayes loss wrt alpha
alpha_t = torch.tensor([[2.0, 3.0]], requires_grad=True)
y = torch.tensor([1])
S_t = alpha_t.sum(1, keepdim=True)
loss = (torch.digamma(S_t) - torch.digamma(alpha_t))[0, 1]
loss.backward()
expected = np.array([polygamma(1, 5.0), polygamma(1, 5.0) - polygamma(1, 3.0)])
check("dL_Bayes/dalpha = psi1(S) - y psi1(alpha)", np.allclose(alpha_t.grad.numpy()[0], expected, atol=1e-5))

# 3. Epistemic mutual information formula against Monte Carlo
alpha = torch.tensor([[2.0, 5.0]])
mi = compute_epistemic_mutual_information(alpha).item()
p = rng.dirichlet([2.0, 5.0], 400000)
H = lambda q: -(q * np.log(q)).sum(1)
mi_mc = H(p.mean(0, keepdims=True))[0] - H(p).mean()
check("Epistemic MI formula vs Monte Carlo", abs(mi - mi_mc) < 5e-3, f"{mi:.4f} vs {mi_mc:.4f}")

# 4. Poisson-subsampled Gaussian RDP accountant against Opacus (if installed)
try:
    from opacus.accountants.analysis.rdp import compute_rdp as op_rdp
    orders = list(range(2, 40))
    ours = np.array([compute_rdp(0.01, 1.1, a) for a in orders])
    ref = np.array(op_rdp(q=0.01, noise_multiplier=1.1, steps=1, orders=orders))
    check("RDP accountant matches Opacus", np.allclose(ours, ref, rtol=1e-6, atol=1e-10), f"max rel err {np.max(np.abs(ours-ref)/ref):.2e}")
except ImportError:
    print("SKIP RDP accountant vs Opacus (opacus not installed)")

# 5. PC-PAT: per-sample gradient of record i does not depend on the other records
from torch.func import functional_call, vmap, grad
model = EvidentialNet(5, 16, 2)
params = {k: v.detach() for k, v in model.named_parameters()}
def loss_i(params, x_i, y_i):
    ev, al, pr, un = functional_call(model, params, (x_i.unsqueeze(0),))
    enc = {k.replace('encoder.', ''): v for k, v in params.items() if k.startswith('encoder.')}
    h = functional_call(model.encoder, enc, (x_i.unsqueeze(0),))
    proto = torch.sum(params['prototypes'] * (torch.arange(2) == y_i).float().unsqueeze(1), dim=0)
    return (edl_bayes_loss(y_i.unsqueeze(0), al) + 0.5 * 0.5 * torch.sum((h.squeeze(0) - proto) ** 2)).sum()
X = torch.randn(8, 5); Y = torch.randint(0, 2, (8,))
g_full = vmap(grad(loss_i), in_dims=(None, 0, 0))(params, X, Y)
g_sub = vmap(grad(loss_i), in_dims=(None, 0, 0))(params, X[:4], Y[:4])
dev = max((g_full[k][:4] - g_sub[k]).abs().max().item() for k in g_full)
check("PC-PAT per-sample gradients independent of other records", dev < 1e-6, f"max dev {dev:.2e}")

# 6. W2 bound of Proposition 2 in one dimension (W2 between empirical measures = RMS of sorted differences)
h0 = rng.normal(0.3, 0.5, 4000); h1 = rng.normal(-0.2, 0.7, 4000); mu = 0.0
w2 = np.sqrt(np.mean((np.sort(h0) - np.sort(h1)) ** 2))
bound = np.sqrt(np.mean((h0 - mu) ** 2)) + np.sqrt(np.mean((h1 - mu) ** 2))
check("W2 <= sqrt(2 E l_PAT) + sqrt(2 E l_PAT)", w2 <= bound + 1e-9, f"{w2:.3f} <= {bound:.3f}")

# 7. DCDA completeness: sum of attributions equals f(x) - f(baseline)
torch.manual_seed(0)
net = EvidentialNet(8, 32, 2); net.eval()
x = torch.randn(64, 8); base = torch.zeros_like(x)
for steps in (30, 50, 200):
    ex = DualChannelDirichletExplainer(net, steps=steps)
    ra, da = ex.attribute(x, baseline=base)
    with torch.no_grad():
        _, al_x, pr_x, _ = net(x); _, al_b, pr_b, _ = net(base)
        dr = (pr_x[:, 1] - pr_b[:, 1]).numpy()
        dd = (compute_epistemic_mutual_information(al_x) - compute_epistemic_mutual_information(al_b)).numpy()
    er = np.abs(ra.sum(1) - dr).max(); ed = np.abs(da.sum(1) - dd).max()
    print(f"     DCDA completeness residual, {steps} Gauss-Legendre nodes: channel A {er:.2e}, channel B {ed:.2e}")
check("DCDA completeness residual < 1e-4 at 50 nodes (claimed in the earlier manuscript)", er < 1e-4 and ed < 1e-4, f"measured {er:.1e} / {ed:.1e}; kinks of LeakyReLU make the integrand non-smooth")
check("DCDA completeness residual < 5e-3 at 50 nodes", er < 5e-3 and ed < 5e-3)

# 8. DP-GCCP coverage guarantee (Theorem 3) by simulation
def coverage_trials(n, mode, trials=300, eps=1.0, a=0.10):
    cov = []
    for _ in range(trials):
        sc = rng.beta(2, 6, n)                        # calibration scores
        cp = DPGroupConditionalConformalPredictor(alpha=a, epsilon_conf=eps, beta=0.05, mode=mode)
        q = cp._quantile(sc, eps, True)
        cov.append(np.mean(rng.beta(2, 6, 20000) <= q))
    return np.array(cov)
for n in (200, 1500, 4000):
    c = coverage_trials(n, "pac", 150)
    ok = np.mean(c >= 0.90) >= 0.95
    check(f"PAC mode: coverage >= 0.90 with prob >= 0.95, n={n}", ok, f"frac ok {np.mean(c >= 0.90):.3f}, mean cov {c.mean():.3f}")
    c = coverage_trials(n, "plugin", 150)
    print(f"     plug-in mode n={n}: mean cov {c.mean():.3f}, frac >= 0.90 {np.mean(c >= 0.90):.3f}")

print("\nSummary:", sum(results.values()), "of", len(results), "checks passed")
