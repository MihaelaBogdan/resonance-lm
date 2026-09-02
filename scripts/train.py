"""Antrenarea RezoNet de la zero (initializare aleatoare, fara greutati imprumutate)."""
import argparse, os, sys, time, json

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import numpy as np

from rezonet.model import RezoNet, RezoConfig
from rezonet.optim import AdamW, cosine_lr
from rezonet.data import load_corpus, Batcher

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def evaluate(model, batcher, max_batches=8):
    tot, n = 0.0, 0
    for i, (x, y) in enumerate(batcher.sequential(limit=max_batches)):
        tot += float(model.loss(x, y).data)
        n += 1
        if n >= max_batches:
            break
    return tot / max(n, 1)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--corpus", default=os.path.join(ROOT, "corpus", "ro.txt"))
    ap.add_argument("--steps", type=int, default=1200)
    ap.add_argument("--batch", type=int, default=24)
    ap.add_argument("--block", type=int, default=128)
    ap.add_argument("--d_model", type=int, default=128)
    ap.add_argument("--layers", type=int, default=3)
    ap.add_argument("--d_bind", type=int, default=160)
    ap.add_argument("--lr", type=float, default=4e-3)
    ap.add_argument("--eval_every", type=int, default=100)
    ap.add_argument("--no_unbind", action="store_true",
                    help="ablatie: dezactiveaza ramura de dezlegare holografica")
    ap.add_argument("--out", default=os.path.join(ROOT, "checkpoints", "rezonet"))
    args = ap.parse_args()

    tok, train_ids, val_ids = load_corpus(args.corpus)
    tok.save(args.out + ".vocab.json")
    print(f"corpus  : {len(train_ids):,} tokeni antrenare / {len(val_ids):,} validare")
    print(f"vocabular: {tok.vocab_size} caractere")

    cfg = RezoConfig(vocab_size=tok.vocab_size, d_model=args.d_model,
                     n_layers=args.layers, d_bind=args.d_bind,
                     tau_max=float(args.block) * 8,
                     unbind=not args.no_unbind)
    model = RezoNet(cfg)
    print(f"model    : {model.num_params():,} parametri, {cfg.n_layers} blocuri, "
          f"d_model={cfg.d_model}, dezlegare={cfg.unbind}")
    print(f"           entropie uniforma = {np.log(tok.vocab_size):.3f} nats/car\n")

    opt = AdamW(model.parameters(), lr=args.lr, weight_decay=0.01, clip=1.0)
    tr = Batcher(train_ids, args.batch, args.block, seed=0)
    va = Batcher(val_ids, args.batch, args.block, seed=1)

    hist = []
    best = float("inf")
    t0 = time.time()
    for step in range(args.steps):
        x, y = tr()
        model.zero_grad()
        loss = model.loss(x, y)
        loss.backward()
        lr = cosine_lr(step, args.steps, args.lr)
        gn = opt.step(lr)

        if step % 20 == 0 or step == args.steps - 1:
            el = time.time() - t0
            print(f"pas {step:5d} | pierdere {float(loss.data):.4f} | "
                  f"lr {lr:.2e} | |g| {gn:5.2f} | {el:6.1f}s", flush=True)

        if (step + 1) % args.eval_every == 0 or step == args.steps - 1:
            vl = evaluate(model, va)
            bpc = vl / np.log(2)
            print(f"        -> validare {vl:.4f} nats = {bpc:.4f} biti/caracter",
                  flush=True)
            hist.append({"step": step + 1, "val": vl, "bpc": bpc})
            if vl < best:
                best = vl
                model.save(args.out + ".npz")
                with open(args.out + ".cfg.json", "w") as f:
                    json.dump(cfg.__dict__, f)

    print(f"\ncea mai buna validare: {best:.4f} nats = {best/np.log(2):.4f} biti/caracter")
    print(f"salvat in {args.out}.npz  ({time.time()-t0:.0f}s total)")
    with open(args.out + ".history.json", "w") as f:
        json.dump(hist, f, indent=1)


if __name__ == "__main__":
    main()
