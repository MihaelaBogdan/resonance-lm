"""
Generare in flux: un token o data, memorie constanta.

RezoNet nu re-citeste contextul la fiecare pas (asa cum face atentia). Intreg
trecutul traieste in starea oscilatoarelor, deci costul per token nu creste
niciodata, indiferent de cate caractere s-au generat deja.
"""
import argparse, json, os, sys, time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import numpy as np

from rezonet.model import RezoNet, RezoConfig
from rezonet.tokenizer import CharTokenizer

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def load(prefix):
    with open(prefix + ".cfg.json") as f:
        cfg = RezoConfig(**json.load(f))
    model = RezoNet(cfg).load(prefix + ".npz")
    tok = CharTokenizer.load(prefix + ".vocab.json")
    return model, tok


def sample_next(logits, temperature=0.8, top_k=0, rng=None):
    z = logits.astype(np.float64) / max(temperature, 1e-6)
    if top_k:
        cut = np.partition(z, -top_k)[-top_k]
        z = np.where(z < cut, -np.inf, z)
    z -= z.max()
    p = np.exp(z)
    p /= p.sum()
    return int(rng.choice(len(p), p=p))


def generate(model, tok, prompt, n=400, temperature=0.8, top_k=0, seed=0):
    rng = np.random.default_rng(seed)
    state = model.init_state()
    ids = tok.encode(prompt)
    logits = None
    for t in ids:                      # incarcare context ("prefill")
        logits = model.step(int(t), state)
    out = []
    for _ in range(n):
        nxt = sample_next(logits, temperature, top_k, rng)
        out.append(nxt)
        logits = model.step(nxt, state)
    return prompt + tok.decode(out)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", default=os.path.join(ROOT, "checkpoints", "rezonet"))
    ap.add_argument("--prompt", default="Dimineața, ")
    ap.add_argument("--n", type=int, default=400)
    ap.add_argument("--temperature", type=float, default=0.8)
    ap.add_argument("--top_k", type=int, default=0)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--bench", action="store_true",
                    help="masoara costul per token la context scurt vs lung")
    a = ap.parse_args()

    model, tok = load(a.ckpt)

    if a.bench:
        state = model.init_state()
        for t in tok.encode("Dimineața, "):
            model.step(int(t), state)
        marks = []
        for block in range(6):
            t0 = time.time()
            for _ in range(200):
                model.step(int(np.random.randint(tok.vocab_size)), state)
            marks.append(((block + 1) * 200, (time.time() - t0) / 200 * 1e3))
        print("context generat | ms per token")
        for ctx, ms in marks:
            print(f"{ctx:>13} | {ms:.3f}")
        print("\n(constant: costul nu creste cu lungimea contextului)")
        return

    print(generate(model, tok, a.prompt, a.n, a.temperature, a.top_k, a.seed))


if __name__ == "__main__":
    main()
