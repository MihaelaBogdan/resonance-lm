"""
Cele doua nuclee originale ale RezoNet, cu forward si backward scrise manual.

1. osc_scan  — banca de oscilatoare amortizate cu decadere selectiva.
2. circconv  — legare holografica prin convolutie circulara (algebra vectorilor
               distribuiti), inlocuitorul multiplicativ al atentiei.

Ambele sunt implementate ca operatii atomice in graf: nu se desfasoara in mii
de noduri, deci memoria ramane O(T*D) in loc de O(T * numar_operatii).
"""
from __future__ import annotations

import numpy as np

from . import autograd
from .autograd import Tensor


def osc_scan(u: Tensor, v: Tensor, rho: Tensor, cosw: Tensor, sinw: Tensor) -> Tensor:
    """
    Recurenta rezonanta. Starea fiecarui canal k este un oscilator complex
    s_t = a_t + i*b_t care se roteste cu unghiul w_k si se stinge cu rata rho:

        s_t = rho_t * e^{i*w} * s_{t-1} + (u_t + i*v_t)

    Rotatia e liniara in stare, deci recurenta e stabila si nu explodeaza.
    Frecventa w_k face fiecare canal sa fie acordat pe un anumit ritm al
    secventei; faza acumulata codifica implicit *cand* a intrat informatia,
    deci modelul nu are nevoie de embedding-uri de pozitie.

    u, v, rho: (B, T, D)   cosw, sinw: (D,)
    intoarce: (B, T, 2D) — partea reala concatenata cu cea imaginara
    """
    U, V, R = u.data, v.data, rho.data
    cw, sw = cosw.data, sinw.data
    B, T, D = U.shape

    dt = autograd.DTYPE
    A = np.empty((B, T, D), dtype=dt)
    Bi = np.empty((B, T, D), dtype=dt)
    a = np.zeros((B, D), dtype=dt)
    b = np.zeros((B, D), dtype=dt)

    for t in range(T):
        r = R[:, t]
        na = r * (a * cw - b * sw) + U[:, t]
        nb = r * (a * sw + b * cw) + V[:, t]
        a, b = na, nb
        A[:, t] = a
        Bi[:, t] = b

    out = Tensor(np.concatenate([A, Bi], axis=-1),
                 _parents=(u, v, rho, cosw, sinw), _op="osc_scan")

    def _bw(g):
        ga_in, gb_in = g[..., :D], g[..., D:]
        gu = np.zeros_like(U)
        gv = np.zeros_like(V)
        gr = np.zeros_like(R)
        gcw = np.zeros_like(cw)
        gsw = np.zeros_like(sw)
        dt = autograd.DTYPE
        da = np.zeros((B, D), dtype=dt)
        db = np.zeros((B, D), dtype=dt)
        zero = np.zeros((B, D), dtype=dt)

        for t in range(T - 1, -1, -1):
            da = da + ga_in[:, t]
            db = db + gb_in[:, t]
            gu[:, t] = da
            gv[:, t] = db

            ap = A[:, t - 1] if t > 0 else zero
            bp = Bi[:, t - 1] if t > 0 else zero
            r = R[:, t]

            gr[:, t] = da * (ap * cw - bp * sw) + db * (ap * sw + bp * cw)
            gcw += (da * r * ap + db * r * bp).sum(axis=0)
            gsw += (-da * r * bp + db * r * ap).sum(axis=0)

            da, db = r * (da * cw + db * sw), r * (-da * sw + db * cw)

        u._accum(gu)
        v._accum(gv)
        rho._accum(gr)
        cosw._accum(gcw)
        sinw._accum(gsw)

    out._backward = _bw
    return out


def circconv(p: Tensor, q: Tensor, scale: float = 1.0) -> Tensor:
    """
    Legare holografica: y = (p (*) q) / sqrt(D), convolutie circulara pe ultima axa.

    Este operatia de "binding" din algebra vectorilor distribuiti: leaga doua
    reprezentari intr-una singura, de aceeasi dimensiune, din care oricare
    poate fi recuperata prin corelatie cu cealalta. Amesteca *toate* perechile
    de trasaturi in O(D log D), nu O(D^2) ca o matrice deasa, si o face
    multiplicativ — de aici capacitatea de a exprima conditionari, nu doar sume.
    """
    D = p.shape[-1]
    P = np.fft.rfft(p.data, n=D, axis=-1)
    Q = np.fft.rfft(q.data, n=D, axis=-1)
    y = np.fft.irfft(P * Q, n=D, axis=-1).astype(autograd.DTYPE) * scale
    out = Tensor(y, _parents=(p, q), _op="circconv")

    def _bw(g):
        G = np.fft.rfft(g.astype(np.float64) * scale, n=D, axis=-1)
        gp = np.fft.irfft(G * np.conj(Q), n=D, axis=-1).astype(autograd.DTYPE)
        gq = np.fft.irfft(G * np.conj(P), n=D, axis=-1).astype(autograd.DTYPE)
        p._accum(gp)
        q._accum(gq)

    out._backward = _bw
    return out


def rms_norm(x: Tensor, gain: Tensor, eps: float = 1e-5) -> Tensor:
    """Normalizare RMS: scaleaza fiecare vector la norma unitara, apoi un castig invatat."""
    ms = (x * x).mean(axis=-1, keepdims=True)
    inv = Tensor(1.0) / (ms + eps).sqrt()
    return x * inv * gain


def circcorr(p: Tensor, q: Tensor, scale: float = 1.0) -> Tensor:
    """
    Dezlegare holografica: y = p (*) q~, unde q~ este involutia lui q
    (q~[n] = q[-n mod D]). In domeniul Fourier: Y = P * conj(Q).

    Este operatia inversa a lui `circconv` si, in algebra vectorilor
    distribuiti, este *interogarea*: extrage din suprapunere componenta
    asociata cu o cheie. Rezultatul la deplasarea 0 este produsul scalar, iar
    la celelalte deplasari este raspunsul unui filtru adaptat — adica un
    intreg spectru de similaritati, calculat in O(D log D).

    Fara ea, blocul poate doar sa combine trasaturi; cu ea, poate sa *caute*
    in starea suprapusa a oscilatoarelor.
    """
    D = p.shape[-1]
    P = np.fft.rfft(p.data, n=D, axis=-1)
    Q = np.fft.rfft(q.data, n=D, axis=-1)
    y = np.fft.irfft(P * np.conj(Q), n=D, axis=-1).astype(autograd.DTYPE) * scale
    out = Tensor(y, _parents=(p, q), _op="circcorr")

    def _bw(g):
        G = np.fft.rfft(g.astype(np.float64) * scale, n=D, axis=-1)
        gp = np.fft.irfft(G * Q, n=D, axis=-1).astype(autograd.DTYPE)
        gq = np.fft.irfft(np.conj(G) * P, n=D, axis=-1).astype(autograd.DTYPE)
        p._accum(gp)
        q._accum(gq)

    out._backward = _bw
    return out
