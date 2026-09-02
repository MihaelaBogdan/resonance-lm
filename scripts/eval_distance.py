"""
Separa doua explicatii posibile ale esecului la distanta mare:
  (a) modelul nu stapaneste formele de plural, sau
  (b) modelul nu duce informatia peste ~95 de caractere.
Acelasi tip de acord (gen la plural), doua distante diferite.
"""
import argparse, os, sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import numpy as np
from make_corpus import NOUNS, ADJS, LOCS, TIMES, adj_for
from eval_agreement import log_probs, load_model

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def probes(kind, rng, corpus, n=250):
    out = []
    guard = 0
    while len(out) < n and guard < 200000:
        guard += 1
        nn = NOUNS[rng.integers(len(NOUNS))]; a = ADJS[rng.integers(len(ADJS))]
        opp = [x for x in NOUNS if x[4] != nn[4]]; d = opp[rng.integers(len(opp))]
        g, b = adj_for(a, nn[4], True), adj_for(a, d[4], True)
        if g == b:
            continue
        if kind == "scurt":
            pre, cand = f"{nn[3].capitalize()} ", (g + " ", b + " ")
        else:
            loc = LOCS[rng.integers(len(LOCS))]; loc2 = LOCS[rng.integers(len(LOCS))]
            tm = TIMES[rng.integers(len(TIMES))].lower()
            pre = (f"{nn[3].capitalize()}, despre care {d[1]} a vorbit {loc} {tm} "
                   f"și {d[3]} au tăcut {loc2}, sunt ")
            cand = (g + ".", b + ".")
        # la distanta scurta, orice pereche substantiv+adjectiv apare in corpus:
        # acolo intrebarea nu e generalizarea, ci daca modelul nimereste deloc.
        if kind == "scurt" or pre + cand[0] not in corpus:
            out.append((pre, *cand))
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", nargs="+", required=True)
    a = ap.parse_args()
    cdir = os.path.join(ROOT, "corpus")
    corpus = "".join(open(os.path.join(cdir, f), encoding="utf-8").read()
                     for f in sorted(os.listdir(cdir)) if f.endswith(".txt"))

    print("ACELASI ACORD (gen la plural), DOUA DISTANTE\n")
    for ck in a.ckpt:
        model, tok = load_model(ck)
        print(f"  {os.path.basename(ck)}:", flush=True)
        for kind in ("scurt", "lung"):
            ps = probes(kind, np.random.default_rng(9), corpus)
            items = []
            for pre, good, bad in ps:
                for c in (good, bad):
                    items.append((pre + c, len(pre), len(pre + c)))
            sc = log_probs(model, tok, items).reshape(-1, 2)
            acc = float((sc[:, 0] > sc[:, 1]).mean())
            ci = 1.96 * np.sqrt(acc * (1 - acc) / len(ps))
            dist = int(np.mean([len(p[0]) for p in ps]))
            print(f"     distanta ~{dist:>3} caractere : {acc*100:5.1f}% +- {ci*100:.1f}"
                  f"   (n={len(ps)})", flush=True)
        print(flush=True)


if __name__ == "__main__":
    main()
