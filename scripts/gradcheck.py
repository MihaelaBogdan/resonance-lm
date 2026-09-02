"""Verificare numerica a gradientilor: derivate analitice vs. diferente finite."""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import rezonet.autograd as ag
ag.set_dtype(np.float64)   # diferentele finite au nevoie de precizie dubla
from rezonet.autograd import Tensor, cross_entropy
from rezonet.ops import osc_scan, circconv, circcorr

rng = np.random.default_rng(0)


def numeric_grad(f, t, eps=1e-6):
    g = np.zeros_like(t.data, dtype=np.float64)
    flat = t.data.reshape(-1)
    for i in range(flat.size):
        orig = flat[i]
        flat[i] = orig + eps; hi = float(f().data)
        flat[i] = orig - eps; lo = float(f().data)
        flat[i] = orig
        g.reshape(-1)[i] = (hi - lo) / (2 * eps)
    return g


def check(name, f, params):
    for p in params:
        p.grad = None
    out = f()
    out.backward()
    worst = 0.0
    for p in params:
        ana = p.grad if p.grad is not None else np.zeros_like(p.data)
        num = numeric_grad(f, p)
        denom = np.maximum(np.abs(ana) + np.abs(num), 1e-3)
        worst = max(worst, float(np.max(np.abs(ana - num) / denom)))
    status = "OK " if worst < 1e-5 else "ESEC"
    print(f"  [{status}] {name:<28} eroare relativa max = {worst:.2e}")
    return worst < 1e-5


ok = True
B, T, D = 2, 5, 4

# --- osc_scan ---
u = Tensor(rng.normal(size=(B, T, D)) * .5, requires_grad=True)
v = Tensor(rng.normal(size=(B, T, D)) * .5, requires_grad=True)
rr = Tensor(rng.uniform(.3, .9, size=(B, T, D)), requires_grad=True)
th = Tensor(rng.uniform(0, 2, size=(D,)), requires_grad=True)
w = Tensor(rng.normal(size=(2 * D, 3)) * .5, requires_grad=True)
ok &= check("osc_scan", lambda: ((osc_scan(u, v, rr, th.cos(), th.sin()) @ w).tanh()).sum(),
            [u, v, rr, th, w])

# --- circconv ---
p = Tensor(rng.normal(size=(B, T, D)), requires_grad=True)
q = Tensor(rng.normal(size=(B, T, D)), requires_grad=True)
ok &= check("circconv", lambda: (circconv(p, q, 1 / np.sqrt(D)) * Tensor(rng.normal(size=(B, T, D)) * 0 + 1.3)).sum(), [p, q])

p2 = Tensor(rng.normal(size=(B, T, D)), requires_grad=True)
q2 = Tensor(rng.normal(size=(B, T, D)), requires_grad=True)
coef = Tensor(rng.normal(size=(B, T, D)))
ok &= check("circcorr", lambda: (circcorr(p2, q2, 1 / np.sqrt(D)) * coef).sum(), [p2, q2])

# --- cross_entropy + embedding + rms_norm, prin modelul intreg ---
from rezonet.model import RezoNet, RezoConfig
cfg = RezoConfig(vocab_size=7, d_model=8, n_layers=2, d_bind=8, seed=1)
m = RezoNet(cfg)
idx = rng.integers(0, 7, size=(2, 6))
tgt = rng.integers(0, 7, size=(2, 6))
ps = m.parameters()
ok &= check("RezoNet (toti parametrii)", lambda: m.loss(idx, tgt), ps)

print("\nTOATE VERIFICARILE AU TRECUT" if ok else "\nVERIFICARE ESUATA")
sys.exit(0 if ok else 1)
