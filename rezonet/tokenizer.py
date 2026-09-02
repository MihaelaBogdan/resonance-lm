"""Tokenizator pe caractere, construit din corpus. Fara vocabular imprumutat."""
from __future__ import annotations

import json

import numpy as np


class CharTokenizer:
    def __init__(self, chars):
        self.itos = list(chars)
        self.stoi = {c: i for i, c in enumerate(self.itos)}

    @classmethod
    def from_text(cls, text):
        return cls(sorted(set(text)))

    @property
    def vocab_size(self):
        return len(self.itos)

    def encode(self, s):
        unk = self.stoi.get(" ", 0)
        return np.array([self.stoi.get(c, unk) for c in s], dtype=np.int64)

    def decode(self, ids):
        return "".join(self.itos[int(i)] for i in ids)

    def save(self, path):
        with open(path, "w", encoding="utf-8") as f:
            json.dump(self.itos, f, ensure_ascii=False)

    @classmethod
    def load(cls, path):
        with open(path, encoding="utf-8") as f:
            return cls(json.load(f))
