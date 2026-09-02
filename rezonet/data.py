"""Incarcarea corpusului si esantionarea loturilor."""
from __future__ import annotations

import numpy as np

from .tokenizer import CharTokenizer


def load_corpus(path, val_frac=0.05):
    with open(path, encoding="utf-8") as f:
        text = f.read()
    tok = CharTokenizer.from_text(text)
    ids = tok.encode(text)
    n_val = max(1024, int(len(ids) * val_frac))
    return tok, ids[:-n_val], ids[-n_val:]


class Batcher:
    def __init__(self, ids, batch_size, block_size, seed=0):
        self.ids = ids
        self.B = batch_size
        self.T = block_size
        self.rng = np.random.default_rng(seed)

    def __call__(self):
        hi = len(self.ids) - self.T - 1
        i = self.rng.integers(0, hi, size=self.B)
        x = np.stack([self.ids[j:j + self.T] for j in i])
        y = np.stack([self.ids[j + 1:j + 1 + self.T] for j in i])
        return x, y

    def sequential(self, limit=None):
        """Ferestre care nu se suprapun — pentru evaluare deterministica."""
        n = (len(self.ids) - 1) // self.T
        if limit:
            n = min(n, limit * self.B)
        for k in range(0, n, self.B):
            idxs = [(k + j) * self.T for j in range(self.B) if (k + j) < n]
            if not idxs:
                break
            x = np.stack([self.ids[j:j + self.T] for j in idxs])
            y = np.stack([self.ids[j + 1:j + 1 + self.T] for j in idxs])
            yield x, y
