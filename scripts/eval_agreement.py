"""
Test lingvistic: a invatat modelul ACORDUL, sau doar statistici de caractere?

Pentru fiecare sonda comparam probabilitatea formei corecte a adjectivului cu
cea a formei gresite (gen sau numar schimbat). Sansa oarba = 50%.

Regula importanta: pastram doar sondele al caror text NU apare in corpus, deci
masuram generalizare la combinatii nevazute, nu memorare.
"""
import argparse, os, sys, json
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import numpy as np
from make_corpus import NOUNS, ADJS, VERBS, LOCS, TIMES, adj_for
from rezonet.model import RezoNet, RezoConfig
from rezonet.tokenizer import CharTokenizer

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
CKPT = os.path.join(ROOT, "checkpoints", "rezonet")


def log_probs(model, tok, items, batch=32):
    """items: (text, start, end) -> log P(text[start:end] | text[:start])"""
    scores = np.zeros(len(items))
    by_len = {}
    for i, (txt, s, e) in enumerate(items):
        by_len.setdefault(len(txt), []).append(i)
    for _, idxs in by_len.items():
        for k in range(0, len(idxs), batch):
            chunk = idxs[k:k + batch]
            X = np.stack([tok.encode(items[i][0]) for i in chunk])
            lg = model.forward(X[:, :-1]).data
            lg = lg - lg.max(-1, keepdims=True)
            lp = lg - np.log(np.exp(lg).sum(-1, keepdims=True))
            for r, i in enumerate(chunk):
                _, s, e = items[i]
                tgt = X[r, s:e]
                scores[i] = lp[r, np.arange(s - 1, e - 1), tgt].sum()
    return scores


def build(rng, corpus, n=500):
    """Trei familii de sonde, de la acord local la acord peste distractor."""
    fams = {"local (adjectiv lipit de substantiv)": [],
            "distant (peste o relativa + substantiv distractor)": [],
            "gen la plural, peste distractor": []}
    guard = 0
    while min(len(v) for v in fams.values()) < n and guard < 60000:
        guard += 1
        nn = NOUNS[rng.integers(len(NOUNS))]
        a = ADJS[rng.integers(len(ADJS))]
        v = VERBS[rng.integers(len(VERBS))]
        loc = LOCS[rng.integers(len(LOCS))]
        opp = [x for x in NOUNS if x[4] != nn[4]]
        d = opp[rng.integers(len(opp))]

        good_sg, bad_sg = adj_for(a, nn[4], False), adj_for(a, d[4], False)
        good_pl = adj_for(a, nn[4], True)
        if good_sg != bad_sg:
            pre = f"{nn[1].capitalize()} "
            if pre + good_sg not in corpus:
                fams["local (adjectiv lipit de substantiv)"].append(
                    (pre, good_sg + " ", bad_sg + " "))
            pre = (f"{nn[1].capitalize()}, despre care {d[1]} a vorbit {loc}, este ")
            if pre + good_sg not in corpus:
                fams["distant (peste o relativa + substantiv distractor)"].append(
                    (pre, good_sg + ".", bad_sg + "."))
        bad_pl = adj_for(a, d[4], True)
        if good_pl != bad_pl:
            tm = TIMES[rng.integers(len(TIMES))].lower()
            loc2 = LOCS[rng.integers(len(LOCS))]
            # tiparul exact folosit in corpus, ca sonda sa fie in-distributie
            pre = (f"{nn[3].capitalize()}, despre care {d[1]} a vorbit {loc} {tm} "
                   f"și {d[3]} au tăcut {loc2}, sunt ")
            if pre + good_pl not in corpus:
                fams["gen la plural, peste distractor"].append(
                    (pre, good_pl + ".", bad_pl + "."))
    return {k: v[:n] for k, v in fams.items()}


def load_model(prefix):
    with open(prefix + ".cfg.json") as f:
        cfg = RezoConfig(**json.load(f))
    return RezoNet(cfg).load(prefix + ".npz"), CharTokenizer.load(prefix + ".vocab.json")


def score_family(model, tok, probes):
    items, meta = [], []
    for pre, good, bad in probes:
        for cont in (good, bad):
            items.append((pre + cont, len(pre), len(pre + cont)))
        meta.append(len(pre))
    sc = log_probs(model, tok, items).reshape(-1, 2)
    win = sc[:, 0] > sc[:, 1]
    acc = float(win.mean())
    n = len(win)
    ci = 1.96 * np.sqrt(acc * (1 - acc) / n)      # interval normal 95%
    return acc, float(ci), int(np.mean(meta))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", nargs="+", default=[CKPT],
                    help="unul sau mai multe prefixe de checkpoint, de comparat")
    a = ap.parse_args()

    cdir = os.path.join(ROOT, "corpus")
    corpus = "".join(open(os.path.join(cdir, f), encoding="utf-8").read()
                     for f in sorted(os.listdir(cdir)) if f.endswith(".txt"))
    fams = build(np.random.default_rng(11), corpus)

    print("TEST DE ACORD GRAMATICAL  (sanse oarbe: 50.0%)")
    print("doar combinatii care NU apar in corpusul de antrenare\n")

    results = {}
    for prefix in a.ckpt:
        model, tok = load_model(prefix)
        results[os.path.basename(prefix)] = {
            name: score_family(model, tok, probes) for name, probes in fams.items()}

    names = list(fams)
    w = max(len(n) for n in names) + 2
    hdr = "sonda".ljust(w) + "dist" + "".join(f"{k:>22}" for k in results)
    print(hdr); print("-" * len(hdr))
    for n in names:
        dist = next(iter(results.values()))[n][2]
        row = n.ljust(w) + f"{dist:>4}"
        for k in results:
            acc, ci, _ = results[k][n]
            row += f"{acc*100:>12.1f}% +-{ci*100:4.1f}"
        print(row)
    print("\n(+- = interval de incredere 95%; suprapunerea lor inseamna "
          "diferenta neconcludenta)")


if __name__ == "__main__":
    main()
