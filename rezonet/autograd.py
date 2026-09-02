"""
Motor de autodiferentiere reverse-mode, scris de la zero peste NumPy.

Nu foloseste niciun framework de deep learning. Un Tensor este un array NumPy
plus o muchie in graful de calcul: fiecare operatie inregistreaza o functie
`_backward` care propaga gradientul catre parinti.
"""
from __future__ import annotations

import numpy as np

# Tipul de date global. float32 pentru antrenare, float64 pentru gradcheck.
DTYPE = np.float32


def set_dtype(dt):
    global DTYPE
    DTYPE = dt


def _unbroadcast(grad: np.ndarray, shape: tuple) -> np.ndarray:
    """Reduce `grad` inapoi la `shape`, anuland broadcasting-ul din forward."""
    if grad.shape == shape:
        return grad
    # colapseaza dimensiunile in plus adaugate la stanga
    while grad.ndim > len(shape):
        grad = grad.sum(axis=0)
    # colapseaza dimensiunile care au fost 1 si s-au intins
    for i, dim in enumerate(shape):
        if dim == 1 and grad.shape[i] != 1:
            grad = grad.sum(axis=i, keepdims=True)
    return grad.reshape(shape)


class Tensor:
    __slots__ = ("data", "grad", "requires_grad", "_parents", "_backward", "_op")

    def __init__(self, data, requires_grad=False, _parents=(), _backward=None, _op=""):
        self.data = np.asarray(data, dtype=DTYPE)
        self.requires_grad = requires_grad or any(p.requires_grad for p in _parents)
        self.grad = None
        self._parents = _parents
        self._backward = _backward
        self._op = _op

    # ---------- infrastructura ----------
    @property
    def shape(self):
        return self.data.shape

    @property
    def ndim(self):
        return self.data.ndim

    def __repr__(self):
        return f"Tensor(shape={self.data.shape}, op={self._op or 'leaf'})"

    def zero_grad(self):
        self.grad = None

    def backward(self):
        """Retropropagare dintr-un scalar."""
        assert self.data.size == 1, "backward() se apeleaza doar pe un scalar"
        topo, seen = [], set()

        def build(t):
            stack = [(t, False)]
            while stack:
                node, expanded = stack.pop()
                if expanded:
                    topo.append(node)
                    continue
                if id(node) in seen:
                    continue
                seen.add(id(node))
                stack.append((node, True))
                for p in node._parents:
                    if id(p) not in seen:
                        stack.append((p, False))

        build(self)
        self.grad = np.ones_like(self.data)
        for node in reversed(topo):
            if node._backward is not None and node.grad is not None:
                node._backward(node.grad)

    def _accum(self, g):
        if not self.requires_grad:
            return
        if self.grad is None:
            self.grad = g.astype(DTYPE, copy=True)
        else:
            self.grad += g

    # ---------- operatii elementwise ----------
    def __add__(self, other):
        other = other if isinstance(other, Tensor) else Tensor(other)
        out = Tensor(self.data + other.data, _parents=(self, other), _op="add")

        def _bw(g):
            self._accum(_unbroadcast(g, self.shape))
            other._accum(_unbroadcast(g, other.shape))

        out._backward = _bw
        return out

    def __mul__(self, other):
        other = other if isinstance(other, Tensor) else Tensor(other)
        out = Tensor(self.data * other.data, _parents=(self, other), _op="mul")

        def _bw(g):
            self._accum(_unbroadcast(g * other.data, self.shape))
            other._accum(_unbroadcast(g * self.data, other.shape))

        out._backward = _bw
        return out

    def __neg__(self):
        return self * -1.0

    def __sub__(self, other):
        other = other if isinstance(other, Tensor) else Tensor(other)
        return self + (-other)

    def __rsub__(self, other):
        return (-self) + other

    def __truediv__(self, other):
        other = other if isinstance(other, Tensor) else Tensor(other)
        out = Tensor(self.data / other.data, _parents=(self, other), _op="div")

        def _bw(g):
            self._accum(_unbroadcast(g / other.data, self.shape))
            other._accum(_unbroadcast(-g * self.data / (other.data ** 2), other.shape))

        out._backward = _bw
        return out

    __radd__ = __add__
    __rmul__ = __mul__

    def __matmul__(self, other):
        out = Tensor(self.data @ other.data, _parents=(self, other), _op="matmul")

        def _bw(g):
            ga = g @ np.swapaxes(other.data, -1, -2)
            gb = np.swapaxes(self.data, -1, -2) @ g
            self._accum(_unbroadcast(ga, self.shape))
            other._accum(_unbroadcast(gb, other.shape))

        out._backward = _bw
        return out

    # ---------- functii ----------
    def exp(self):
        e = np.exp(self.data)
        out = Tensor(e, _parents=(self,), _op="exp")
        out._backward = lambda g: self._accum(g * e)
        return out

    def log(self):
        out = Tensor(np.log(self.data), _parents=(self,), _op="log")
        out._backward = lambda g: self._accum(g / self.data)
        return out

    def sqrt(self):
        r = np.sqrt(self.data)
        out = Tensor(r, _parents=(self,), _op="sqrt")
        out._backward = lambda g: self._accum(g * 0.5 / (r + 1e-12))
        return out

    def tanh(self):
        t = np.tanh(self.data)
        out = Tensor(t, _parents=(self,), _op="tanh")
        out._backward = lambda g: self._accum(g * (1 - t * t))
        return out

    def sigmoid(self):
        s = 1.0 / (1.0 + np.exp(-np.clip(self.data, -60, 60)))
        out = Tensor(s, _parents=(self,), _op="sigmoid")
        out._backward = lambda g: self._accum(g * s * (1 - s))
        return out

    def silu(self):
        s = 1.0 / (1.0 + np.exp(-np.clip(self.data, -60, 60)))
        v = self.data * s
        out = Tensor(v, _parents=(self,), _op="silu")
        out._backward = lambda g: self._accum(g * (s + v * (1 - s)))
        return out

    def softplus(self):
        x = self.data
        out = Tensor(np.logaddexp(0.0, x), _parents=(self,), _op="softplus")
        out._backward = lambda g: self._accum(g / (1.0 + np.exp(-np.clip(x, -60, 60))))
        return out

    def cos(self):
        out = Tensor(np.cos(self.data), _parents=(self,), _op="cos")
        out._backward = lambda g: self._accum(-g * np.sin(self.data))
        return out

    def sin(self):
        out = Tensor(np.sin(self.data), _parents=(self,), _op="sin")
        out._backward = lambda g: self._accum(g * np.cos(self.data))
        return out

    # ---------- forma / reduceri ----------
    def sum(self, axis=None, keepdims=False):
        out = Tensor(self.data.sum(axis=axis, keepdims=keepdims), _parents=(self,), _op="sum")

        def _bw(g):
            if axis is not None and not keepdims:
                g = np.expand_dims(g, axis)
            self._accum(np.broadcast_to(g, self.shape).copy())

        out._backward = _bw
        return out

    def mean(self, axis=None, keepdims=False):
        n = self.data.size if axis is None else self.data.shape[axis]
        return self.sum(axis=axis, keepdims=keepdims) * (1.0 / n)

    def reshape(self, *shape):
        if len(shape) == 1 and isinstance(shape[0], (tuple, list)):
            shape = tuple(shape[0])
        old = self.shape
        out = Tensor(self.data.reshape(shape), _parents=(self,), _op="reshape")
        out._backward = lambda g: self._accum(g.reshape(old))
        return out

    def transpose(self, *axes):
        out = Tensor(self.data.transpose(axes), _parents=(self,), _op="transpose")
        inv = np.argsort(axes)
        out._backward = lambda g: self._accum(g.transpose(inv))
        return out

    def slice_last(self, start, stop):
        """Felie pe ultima axa — folosita pentru a sparge o proiectie fuzionata."""
        sl = (Ellipsis, slice(start, stop))
        out = Tensor(self.data[sl], _parents=(self,), _op="slice")

        def _bw(g):
            z = np.zeros_like(self.data)
            z[sl] = g
            self._accum(z)

        out._backward = _bw
        return out


# ---------- operatii cu semnatura speciala ----------

def embedding(table: Tensor, idx: np.ndarray) -> Tensor:
    """Cautare in tabel; backward = scatter-add."""
    out = Tensor(table.data[idx], _parents=(table,), _op="embedding")

    def _bw(g):
        z = np.zeros_like(table.data)
        np.add.at(z, idx, g)
        table._accum(z)

    out._backward = _bw
    return out


def cross_entropy(logits: Tensor, targets: np.ndarray) -> Tensor:
    """Entropie incrucisata medie, stabila numeric. logits (N,V), targets (N,)."""
    x = logits.data
    x = x - x.max(axis=-1, keepdims=True)
    e = np.exp(x)
    p = e / e.sum(axis=-1, keepdims=True)
    n = targets.shape[0]
    loss = -np.log(p[np.arange(n), targets] + 1e-12).mean()
    out = Tensor(np.asarray(loss, dtype=DTYPE), _parents=(logits,), _op="cross_entropy")

    def _bw(g):
        d = p.copy()
        d[np.arange(n), targets] -= 1.0
        logits._accum(g * d / n)

    out._backward = _bw
    return out
