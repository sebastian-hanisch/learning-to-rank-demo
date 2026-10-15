"""Szenario: Reglerwerte (Settings), die Wahrheit (Merkmale → Stärke) und die simulierten Gruppen („Saisons“) aus neuen Spielern.

Wahrheit: Spieler mit p Merkmalen x ~ N(0, I); Stärke s = f(x) + σ_u · u mit u ~ N(0, 1). f linear (|w| = 1) oder nichtlinear (tanh, sin, Wechselwirkung); σ_u ist, was die Merkmale nicht
erklären. Eine Gruppe sind m = 12 neue Spieler; je Paar werden k Partien gespielt (Bradley-Terry mit s), in den Testgruppen mit der Wahrscheinlichkeit q je Paar eine Partie.

Zufall nur über SplitMix64 (Generator der Linie), je Gruppe ein eigener Strom aus (Seed, laufende Nummer): Gruppe g hängt nur von (Seed, g) ab, also sind die ersten Q Gruppen bei größerem
Q dieselben. Anordnung je Gruppe: m·p + m Normalwerte (Merkmale, dann u), 66 Werte für die Auswahl der gespielten Paare, 66·k Werte für die Ausgänge."""

import math
from dataclasses import dataclass

import numpy as np

import ltr_constants as C
import turnier_generator as T

_SALT_TRUTH, _SALT_GROUP = 0x17A1, 0x61B0
_MASK = (1 << 64) - 1


@dataclass(frozen=True)
class Settings:
    p: int = C.DEFAULT_P
    sigma_u: float = C.DEFAULT_SIGMA
    truth: str = C.DEFAULT_TRUTH
    q_groups: int = C.DEFAULT_Q
    k_train: int = C.DEFAULT_K
    loss: str = C.DEFAULT_LOSS
    model: str = C.DEFAULT_MODEL
    q_test: float = C.DEFAULT_Q_TEST
    l2: float = C.DEFAULT_L2
    seed: int = C.DEFAULT_SEED


@dataclass(frozen=True)
class Groups:
    X: np.ndarray               # (Q, m, p)
    s: np.ndarray               # (Q, m) wahre Stärke
    f_true: np.ndarray          # (Q, m) wahre Funktion der Merkmale ohne das Unerklärte
    W: np.ndarray               # (Q, m, m) Siege von i gegen j
    k: int
    q: float


def normals(rng: T.SplitMix64, count: int) -> np.ndarray:
    """Standardnormalverteilte Werte (Box-Muller) aus dem SplitMix64-Strom."""
    half = (count + 1) // 2
    u1 = np.maximum(rng.uniform(half), 1e-300)
    u2 = rng.uniform(half)
    r = np.sqrt(-2.0 * np.log(u1))
    z = np.concatenate([r * np.cos(2.0 * math.pi * u2), r * np.sin(2.0 * math.pi * u2)])
    return z[:count]


@dataclass(frozen=True)
class Truth:
    w: np.ndarray
    v: np.ndarray
    nonlinear: bool
    sigma_u: float

    @property
    def p(self) -> int:
        return len(self.w)

    def f(self, X: np.ndarray) -> np.ndarray:
        if self.nonlinear:
            return (0.8 * np.tanh(2.0 * (X @ self.w)) + 0.6 * np.sin(1.5 * (X @ self.v)) + 0.4 * X[..., 0] * X[..., 1]) / 0.9
        return X @ self.w


def make_truth(p: int, truth: str, sigma_u: float, seed: int) -> Truth:
    z = normals(T.SplitMix64(int(seed) ^ _SALT_TRUTH), 2 * p)
    w, v = z[:p], z[p:]
    return Truth(w / np.linalg.norm(w), v / np.linalg.norm(v), truth == C.TRUTHS[1], float(sigma_u))


def _group_rng(seed: int, index: int) -> T.SplitMix64:
    return T.SplitMix64((int(seed) * 0x9E3779B97F4A7C15 + (int(index) + 1) * 0xBF58476D1CE4E5B9 + _SALT_GROUP) & _MASK)


def make_groups(truth: Truth, count: int, k: int, seed: int, offset: int = 0, q: float = 1.0, m: int = C.M_GROUP) -> Groups:
    """`count` Gruppen mit den laufenden Nummern offset … offset + count − 1: je Paar mit der Wahrscheinlichkeit q gespielt, dann k Partien."""
    p = truth.p
    iu = np.triu_indices(m, 1)
    pairs = len(iu[0])
    X = np.empty((count, m, p))
    s = np.empty((count, m))
    ft = np.empty((count, m))
    W = np.zeros((count, m, m))
    for g in range(count):
        rng = _group_rng(seed, offset + g)
        z = normals(rng, m * p + m)
        x = z[: m * p].reshape(m, p)
        f = truth.f(x)
        sg = f + truth.sigma_u * z[m * p:]
        keep = rng.uniform(pairs) < q
        u = rng.uniform(pairs * max(k, 1)).reshape(max(k, 1), pairs)
        p_win = 1.0 / (1.0 + np.exp(-(sg[iu[0]] - sg[iu[1]])))
        wins = (u[:k] < p_win[None, :]).sum(axis=0) if k > 0 else np.zeros(pairs)
        played = np.where(keep, k, 0)
        W[g, iu[0], iu[1]] = np.where(keep, wins, 0)
        W[g, iu[1], iu[0]] = played - np.where(keep, wins, 0)
        X[g], s[g], ft[g] = x, sg, f
    return Groups(X, s, ft, W, k, q)
