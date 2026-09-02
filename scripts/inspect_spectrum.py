"""
Ce a invatat banca de oscilatoare?

Fiecare canal are o frecventa proprie (perioada = 2*pi/theta, in caractere) si
o constanta de timp (tau = 1/softplus(lam), cate caractere tine minte). Le
comparam cu initializarea: daca antrenarea le-a mutat, inseamna ca modelul si-a
*ales* singur scarile de timp de care avea nevoie.
"""
import argparse, json, os, sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import numpy as np
from rezonet.model import RezoNet, RezoConfig

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def hist(vals, lo, hi, bins=9, width=34):
    edges = np.exp(np.linspace(np.log(lo), np.log(hi), bins + 1))
    cnt, _ = np.histogram(np.clip(vals, lo, hi * 0.999), bins=edges)
    top = max(cnt.max(), 1)
    for i in range(bins):
        bar = "#" * int(round(width * cnt[i] / top))
        print(f"    {edges[i]:8.1f} - {edges[i+1]:8.1f} | {bar} {cnt[i]}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", default=os.path.join(ROOT, "checkpoints", "rezonet"))
    a = ap.parse_args()
    with open(a.ckpt + ".cfg.json") as f:
        cfg = RezoConfig(**json.load(f))
    trained = RezoNet(cfg).load(a.ckpt + ".npz")
    fresh = RezoNet(cfg)          # aceeasi initializare, neantrenata

    for i in range(cfg.n_layers):
        th = trained._params[f"b{i}.theta"].data
        lam = trained._params[f"b{i}.lam"].data
        th0 = fresh._params[f"b{i}.theta"].data
        lam0 = fresh._params[f"b{i}.lam"].data
        per = 2 * np.pi / np.maximum(np.abs(th), 1e-6)
        tau = 1.0 / np.logaddexp(0.0, lam)
        tau0 = 1.0 / np.logaddexp(0.0, lam0)
        print(f"\n--- blocul {i} ---")
        print(f"  perioade invatate (caractere/ciclu), mediana {np.median(per):.1f} "
              f"[init {np.median(2*np.pi/th0):.1f}]")
        hist(per, 2, 4000)
        print(f"  constante de timp tau (caractere), mediana {np.median(tau):.1f} "
              f"[init {np.median(tau0):.1f}], maxim {tau.max():.0f}")
        hist(tau, 1, 4000)
        moved = np.abs(np.log(tau / tau0))
        print(f"  deplasare medie a lui tau fata de initializare: "
              f"x{np.exp(moved.mean()):.2f}")


if __name__ == "__main__":
    main()
