"""
Verifica faptul ca cele doua cai de calcul dau acelasi rezultat:
  (a) trecerea in paralel pe toata secventa, folosita la antrenare;
  (b) trecerea in flux, token cu token, folosita la generare.
Daca ele nu coincid, generarea nu reflecta modelul antrenat.
"""
import os, sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import numpy as np
from rezonet.model import RezoNet, RezoConfig

cfg = RezoConfig(vocab_size=17, d_model=24, n_layers=3, d_bind=32, seed=5)
m = RezoNet(cfg)
rng = np.random.default_rng(0)
idx = rng.integers(0, 17, size=(1, 40))

parallel = m.forward(idx).data[0]
state = m.init_state()
stream = np.stack([m.step(int(t), state) for t in idx[0]])

err = np.abs(parallel - stream).max() / (np.abs(parallel).max() + 1e-9)
print(f"diferenta relativa maxima paralel vs. flux: {err:.2e}")
print("OK — caile coincid" if err < 1e-4 else "ESEC — caile difera")
sys.exit(0 if err < 1e-4 else 1)
