"""AdamW + taiere de gradient + program cosinus, scrise de la zero."""
from __future__ import annotations

import numpy as np


class AdamW:
    def __init__(self, params, lr=3e-3, betas=(0.9, 0.95), eps=1e-8,
                 weight_decay=0.01, clip=1.0):
        self.params = list(params)
        self.lr = lr
        self.b1, self.b2 = betas
        self.eps = eps
        self.wd = weight_decay
        self.clip = clip
        self.t = 0
        self.m = [np.zeros_like(p.data) for p in self.params]
        self.v = [np.zeros_like(p.data) for p in self.params]

    def grad_norm(self):
        s = 0.0
        for p in self.params:
            if p.grad is not None:
                s += float((p.grad.astype(np.float64) ** 2).sum())
        return np.sqrt(s)

    def step(self, lr=None):
        lr = self.lr if lr is None else lr
        self.t += 1
        gn = self.grad_norm()
        scale = min(1.0, self.clip / (gn + 1e-12)) if self.clip else 1.0

        bc1 = 1 - self.b1 ** self.t
        bc2 = 1 - self.b2 ** self.t
        for i, p in enumerate(self.params):
            if p.grad is None:
                continue
            g = p.grad * scale
            self.m[i] = self.b1 * self.m[i] + (1 - self.b1) * g
            self.v[i] = self.b2 * self.v[i] + (1 - self.b2) * (g * g)
            mh = self.m[i] / bc1
            vh = self.v[i] / bc2
            upd = mh / (np.sqrt(vh) + self.eps)
            if self.wd and p.data.ndim >= 2:      # fara decay pe castiguri/biasuri 1D
                upd = upd + self.wd * p.data
            p.data -= (lr * upd).astype(p.data.dtype)
        return gn


def cosine_lr(step, total, base_lr, warmup=100, min_ratio=0.08):
    if step < warmup:
        return base_lr * (step + 1) / warmup
    prog = (step - warmup) / max(1, total - warmup)
    prog = min(1.0, max(0.0, prog))
    return base_lr * (min_ratio + (1 - min_ratio) * 0.5 * (1 + np.cos(np.pi * prog)))
