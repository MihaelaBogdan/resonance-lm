# RezoNet — a language model built from scratch

Sequence model with custom architecture, trained from random initialization.
It does not start from the weights, tokenizer, or structure of any existing model, and uses no machine learning framework: the auto-differentiation engine, optimizer, and computation kernels are written from scratch on top of NumPy.

---

## The Concept

A Transformer remembers by **re-reading** the entire context at every step. RezoNet remembers by **resonating**: each channel is a damped oscillator tuned to its own natural frequency. The current token hits it, and what remains from previous impacts is the memory. The past is not re-read — it keeps vibrating.

A block consists of two stages.

**1. Resonance.** The state of each channel *k* is a complex number that rotates and decays:

```
s_t = ρ_t · e^(i·ω_k) · s_{t-1} + (u_t + i·v_t)
```

- `ω_k` — channel natural frequency, learned, logarithmically initialized.
- `ρ_t = exp(−softplus(λ_k)·(1 + g_t))` — how much the channel retains. The term `g_t` is computed from the current token, so **forgetting is content-dependent**: the model can clear state when the text requires it.
- The recurrence is linear in state, hence stable: `|ρ| < 1` guarantees nothing explodes, no matter how long the sequence is.

Reading uses the real part, imaginary part, **and the envelope** `|s| = √(a²+b²)`.
The envelope is the non-linear part of the stage and is phase-invariant — it responds to *how much* a pattern has resonated, regardless of *where* it started.

**2. Holographic Binding and Unbinding.** Instead of attention and instead of a perceptron, the block uses Vector Symbolic Architecture (distributed vector algebra):

```
bind:      z₁ = p ⊛ q          (circular convolution)
query:     z₂ = p ⊛ q̃         (circular correlation — the inverse operation)
```

Convolution multiplicatively mixes all feature pairs in O(D log D), not O(D²). Correlation is the *query*: it extracts the component associated with a key from the superimposed state. Both are computed via FFT.

### Three Built-In Properties

| | |
|---|---|
| **No position embeddings** | The accumulated phase of each oscillator reveals how long ago a signal entered. Position is not added — it is measured. |
| **Constant generation cost** | A token costs the same whether 10 or 10,000 characters lie behind it. Attention costs O(T). |
| **Multi-scale memory** | Channels start with time constants from ~1 to ~1000 characters; training rearranges them automatically. |

### What Is New and What Is Not

To be honest: **linear recurrence with diagonal complex state** belongs to the same mathematical family as modern state-space models (SSMs), and **binding via circular convolution** comes from Holographic Reduced Representations (Plate, 1995). Neither was invented here.

What is original to this model: combining them in a single block — an explicit oscillator with learned frequency + content-dependent forgetting + envelope readout, followed by a mixer that *both binds and unbinds* — plus the fact that everything, including automatic differentiation, is written from scratch.

---

## Structure

```
rezonet/
  autograd.py    reverse-mode auto-differentiation engine on top of NumPy
  ops.py         osc_scan (resonance), circconv (binding), circcorr (unbinding)
  model.py       architecture + streaming inference path
  optim.py       AdamW, gradient clipping, cosine scheduler
  tokenizer.py   character-level tokenizer built from corpus
  data.py        corpus loading and sampling
scripts/
  make_corpus.py       corpus generator (Romanian grammar with agreement rules)
  train.py             training script
  sample.py            streaming text generation (--bench for per-token latency)
  eval_agreement.py    grammatical agreement benchmark with confidence intervals
  inspect_spectrum.py  inspect learned frequencies and time scales
  gradcheck.py         gradient verification via finite differences
  test_consistency.py  parallel path == streaming path verification
```

## Usage

Corpus generation:

```bash
python3 scripts/make_corpus.py --bytes 1700000 --out corpus/ro_big.txt
```

Training:

```bash
python3 scripts/train.py --corpus corpus/ro_big.txt --steps 4000 --out checkpoints/model
```

Text generation:

```bash
python3 scripts/sample.py --ckpt checkpoints/model --prompt "Dimineața, " --n 400
```

You can train on your own text — any `.txt` file works:

```bash
python3 scripts/train.py --corpus path/to/your_text.txt --steps 4000
```

## Correctness

Two checks that run independently of training:

```bash
python3 scripts/gradcheck.py        # analytical gradients vs. finite differences
python3 scripts/test_consistency.py # training and streaming generation produce identical output
```

Gradcheck compares each hand-derived gradient against finite differences in double precision. The error decreases quadratically with step size `eps` — the hallmark of truncation error, proving the remaining discrepancy is numerical rather than a derivation bug.

---

## Results

All models: 3 blocks, `d_model=128`, ~615,000 parameters, trained on CPU in 7–37 minutes. Vocabulary size of 45 characters; a uniform random guess would cost 5.49 bits/char.

| model | corpus | window | unbinding | bits/char |
|---|---|---|---|---|
| `base` | 390 KB | 128 | no | 0.503 |
| `rezonet` | 390 KB | 128 | **yes** | 0.504 |
| `rezonet_v2` | 1.7 MB | 128 | yes | 0.471 |
| `rezonet_v3` | 1.7 MB | 256 | yes | **0.444** |
| `rezonet_v4` | 1.6 MB* | 256 | yes | 0.387* |

\* `rezonet_v4` was trained on a dataset with a different composition (more frequent long sentences), so its score is **not** directly comparable to the others.

### Grammatical Agreement Benchmark

We measure the log-probability of the correct adjective form versus the incorrect form. Only combinations that do not appear in the training corpus are evaluated, thus measuring true generalization. Chance accuracy is 50%.

| probe | distance | `base` | `rezonet` | `rezonet_v2` | `rezonet_v3` | `rezonet_v4` |
|---|---|---|---|---|---|---|
| adjective adjacent to noun | 7 chars | 96.0% | 97.8% | 100% | 100% | 100% |
| across relative clause + 1 distractor | 53 chars | 57.0% | 63.4% | 99.8% | **100%** | 99.6% |
| across relative clause + 2 distractors | 95 chars | 48.6% | 45.8% | 52.0% | 49.0% | 54.8% |

(95% confidence intervals: ±4.4 percentage points on the last two rows)

**What the table shows.** The 53-character probe is designed to be impossible to solve via local statistics: between the target noun and the adjective sits another noun of opposite gender, while the relative pronoun ("care") carries no gender information. A model relying on local context would systematically fail. RezoNet solves it completely.

**What doesn't work, and why.** At 95 characters with two distractors, the model performs at chance level. We tested four hypotheses:

1. *Insufficient data* — 4x more training text: solved the 53-char case, but not the 95-char case.
2. *Training window too short* — doubled to 256: no effect.
3. *Rare construction* — 3.5x more frequent instances of this specific pattern: no effect.
4. *Inability to handle plural forms* — directly disproved: the **exact same** plural gender agreement achieves 100% at 8 chars and chance at 95.

This indicates a genuine length boundary for the architecture at this model scale, rather than a dataset artifact. It is precisely the kind of limitation this benchmark was built to uncover.

### Learned Spectrum and Time Scales

`scripts/inspect_spectrum.py` reveals the learned time constants. Initialization allowed memory spans up to 1024 characters; when trained with a window size of 128, the model automatically shortened its longest time constant to **262 characters** and clustered its oscillation periods around 25–60 characters — roughly the length of a sentence. It was not instructed to do this; it discovered the time scales of the text on its own.

### Constant Latency During Generation

`python3 scripts/sample.py --bench` measures per-token latency as the context grows:

```
generated context | ms per token
              200 | 0.312
              600 | 0.307
             1200 | 0.308
```

Completely flat. An attention-based model would have scaled ~6x over the same span.

### Generation Example

Prompted with "Pădurea, despre care ":

> Pădurea, despre care copilul a vorbit sub cerul senin, este **frumoasă**.

The head noun ("Pădurea") is feminine, the intervening distractor ("copilul") is masculine, and the adjective 40 characters away is correctly inflected in the feminine.

The semantic meaning is nonsensical because the synthetic corpus itself is nonsensical — its grammar is strict, but its semantics are randomized. The model learned exactly what it was given.

---

## Limitations

- Tested only on a small-scale synthetic corpus (~615,000 parameters). Nothing presented here guarantees behavior at large scale.
- Long-range agreement degrades beyond ~50–60 characters when multiple distractor nouns are introduced.
- Sequential recurrence runs in NumPy. Being linear, it could be parallelized via associative scan — not implemented here.
- No GPU acceleration, no data parallelism.
