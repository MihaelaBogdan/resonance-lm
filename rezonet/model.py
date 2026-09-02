"""
RezoNet — o arhitectura de secventa construita de la zero.

Nu contine atentie, nu contine porti LSTM/GRU si nu porneste de la greutatile
sau structura vreunui model existent. Un bloc are doua etaje:

  1. REZONANTA  — o banca de oscilatoare complexe amortizate, fiecare acordat pe
     alta frecventa si alta constanta de timp. Tokenul curent le "loveste";
     starea retinuta este tiparul de interferenta al loviturilor anterioare.
     Citirea foloseste si anvelopa |s| (detectie de anvelopa), care este partea
     neliniara, invarianta la faza, a blocului.

  2. LEGARE HOLOGRAFICA — convolutie circulara intre doua proiectii ale starii.
     Amesteca multiplicativ toate perechile de trasaturi in O(D log D).

Trei proprietati care rezulta din constructie, nu din trucuri adaugate:
  * fara embedding-uri de pozitie — faza acumulata a fiecarui oscilator spune
    de cat timp a intrat un semnal;
  * cost O(1) per token la generare si memorie constanta, indiferent de cat de
    lung e contextul (atentia cere O(T));
  * timpi de memorie multi-scala, initializati logaritmic de la ~1 pas la ~1000.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from .autograd import Tensor, embedding, cross_entropy
from .ops import osc_scan, circconv, circcorr, rms_norm


@dataclass
class RezoConfig:
    vocab_size: int = 96
    d_model: int = 128
    n_layers: int = 4
    d_bind: int = 192
    seed: int = 1337
    # spectrul de constante de timp la initializare (in pasi de secventa)
    tau_min: float = 1.5
    tau_max: float = 900.0
    # ramura de dezlegare holografica (interogarea starii suprapuse)
    unbind: bool = True


def _inv_softplus(y):
    return np.log(np.expm1(np.clip(y, 1e-7, None)))


class RezoNet:
    def __init__(self, cfg: RezoConfig):
        self.cfg = cfg
        rng = np.random.default_rng(cfg.seed)
        D, L, Db, V = cfg.d_model, cfg.n_layers, cfg.d_bind, cfg.vocab_size
        self._params = {}

        def par(name, arr):
            t = Tensor(arr, requires_grad=True)
            self._params[name] = t
            return t

        def lin(name, fan_in, fan_out, gain=1.0):
            return par(name, rng.normal(0, gain / np.sqrt(fan_in), (fan_in, fan_out)))

        res_gain = 1.0 / np.sqrt(2 * L)  # blocurile reziduale pornesc "mici"

        self.emb = par("emb", rng.normal(0, 0.02, (V, D)))
        self.blocks = []
        for i in range(L):
            b = {}
            b["n1"] = par(f"b{i}.n1", np.ones(D))
            # proiectie fuzionata -> [u | v | poarta_de_uitare]
            b["Win"] = lin(f"b{i}.Win", D, 3 * D)
            # frecvente proprii: log-spatiate, de la lent la rapid
            theta0 = np.exp(np.linspace(np.log(2 * np.pi / cfg.tau_max),
                                        np.log(np.pi * 0.9), D))
            b["theta"] = par(f"b{i}.theta", theta0)
            # amortizare: rho = exp(-softplus(lam)); tau = 1/softplus(lam)
            lam0 = _inv_softplus(1.0 / np.exp(np.linspace(np.log(cfg.tau_min),
                                                          np.log(cfg.tau_max), D)))
            b["lam"] = par(f"b{i}.lam", lam0)
            b["Wa"] = lin(f"b{i}.Wa", D, D, res_gain)
            b["Wb"] = lin(f"b{i}.Wb", D, D, res_gain)
            b["Wm"] = lin(f"b{i}.Wm", D, D, res_gain)

            b["n2"] = par(f"b{i}.n2", np.ones(D))
            b["Wp"] = lin(f"b{i}.Wp", D, Db)
            b["Wq"] = lin(f"b{i}.Wq", D, Db)
            b["Wo"] = lin(f"b{i}.Wo", Db, D, res_gain)
            if cfg.unbind:
                b["Wr"] = lin(f"b{i}.Wr", D, Db)
                b["Wo2"] = lin(f"b{i}.Wo2", Db, D, res_gain)
            self.blocks.append(b)

        self.nf = par("nf", np.ones(D))
        self.head = lin("head", D, V, 0.5)

    # ------------------------------------------------------------------
    def parameters(self):
        return list(self._params.values())

    def num_params(self):
        return sum(p.data.size for p in self._params.values())

    def zero_grad(self):
        for p in self._params.values():
            p.grad = None

    # ------------------------------------------------------------------
    def forward(self, idx: np.ndarray) -> Tensor:
        """idx: (B, T) intregi -> logits (B, T, V)"""
        D = self.cfg.d_model
        x = embedding(self.emb, idx)

        for b in self.blocks:
            # --- etaj 1: rezonanta ---
            h = rms_norm(x, b["n1"])
            proj = h @ b["Win"]
            u = proj.slice_last(0, D)
            v = proj.slice_last(D, 2 * D)
            gate = proj.slice_last(2 * D, 3 * D).softplus()   # uitare suplimentara >= 0
            decay = b["lam"].softplus()                       # rata de baza > 0
            rho = (-(decay * (Tensor(1.0) + gate))).exp()     # in (0, 1)

            s = osc_scan(u, v, rho, b["theta"].cos(), b["theta"].sin())
            a = s.slice_last(0, D)
            im = s.slice_last(D, 2 * D)
            mag = ((a * a + im * im) + Tensor(1e-6)).sqrt()    # anvelopa
            x = x + (a @ b["Wa"]) + (im @ b["Wb"]) + (mag @ b["Wm"])

            # --- etaj 2: legare + dezlegare holografica ---
            h = rms_norm(x, b["n2"])
            sc = 1.0 / np.sqrt(self.cfg.d_bind)
            hp = h @ b["Wp"]
            x = x + (circconv(hp, h @ b["Wq"], sc).silu() @ b["Wo"])   # leaga
            if self.cfg.unbind:
                x = x + (circcorr(hp, h @ b["Wr"], sc).silu() @ b["Wo2"])  # interogheaza

        x = rms_norm(x, self.nf)
        return x @ self.head

    def loss(self, idx: np.ndarray, targets: np.ndarray) -> Tensor:
        logits = self.forward(idx)
        V = self.cfg.vocab_size
        return cross_entropy(logits.reshape(-1, V), targets.reshape(-1))

    # ------------------------------------------------------------------
    # Cale de inferenta in flux: un token o data, memorie constanta.
    # Aceasta e proprietatea pe care atentia nu o are.
    # ------------------------------------------------------------------
    def init_state(self):
        D = self.cfg.d_model
        return [{"a": np.zeros(D, np.float32), "b": np.zeros(D, np.float32)}
                for _ in self.blocks]

    @staticmethod
    def _rms(x, g, eps=1e-5):
        return x / np.sqrt((x * x).mean(-1, keepdims=True) + eps) * g

    def step(self, token: int, state) -> np.ndarray:
        """Un singur token -> logits (V,). Actualizeaza `state` pe loc."""
        P = {k: t.data for k, t in self._params.items()}
        D = self.cfg.d_model
        x = P["emb"][token]

        for i, st in enumerate(state):
            h = self._rms(x, P[f"b{i}.n1"])
            proj = h @ P[f"b{i}.Win"]
            u, v, gpre = proj[:D], proj[D:2 * D], proj[2 * D:]
            gate = np.logaddexp(0.0, gpre)
            decay = np.logaddexp(0.0, P[f"b{i}.lam"])
            rho = np.exp(-decay * (1.0 + gate))
            th = P[f"b{i}.theta"]
            cw, sw = np.cos(th), np.sin(th)

            a0, b0 = st["a"], st["b"]
            a = rho * (a0 * cw - b0 * sw) + u
            bb = rho * (a0 * sw + b0 * cw) + v
            st["a"], st["b"] = a, bb

            mag = np.sqrt(a * a + bb * bb + 1e-6)
            x = x + a @ P[f"b{i}.Wa"] + bb @ P[f"b{i}.Wb"] + mag @ P[f"b{i}.Wm"]

            h = self._rms(x, P[f"b{i}.n2"])
            Db = self.cfg.d_bind
            silu = lambda z: z * (1.0 / (1.0 + np.exp(-np.clip(z, -60, 60))))
            Pf = np.fft.rfft(h @ P[f"b{i}.Wp"], n=Db)
            Qf = np.fft.rfft(h @ P[f"b{i}.Wq"], n=Db)
            z = (np.fft.irfft(Pf * Qf, n=Db) / np.sqrt(Db)).astype(np.float32)
            x = x + silu(z) @ P[f"b{i}.Wo"]
            if self.cfg.unbind:
                Rf = np.fft.rfft(h @ P[f"b{i}.Wr"], n=Db)
                z2 = (np.fft.irfft(Pf * np.conj(Rf), n=Db) / np.sqrt(Db)).astype(np.float32)
                x = x + silu(z2) @ P[f"b{i}.Wo2"]

        x = self._rms(x, P["nf"])
        return x @ P["head"]

    # ------------------------------------------------------------------
    def save(self, path):
        np.savez(path, **{k: v.data for k, v in self._params.items()},
                 __cfg__=np.array(str(self.cfg.__dict__)))

    def load(self, path):
        z = np.load(path, allow_pickle=True)
        for k, t in self._params.items():
            t.data = z[k].astype(np.float32)
        return self
