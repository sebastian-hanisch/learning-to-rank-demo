"""Duellen allein und Hybrid: Bradley-Terry je Gruppe mit dem Prior N(m, 1/λ_u) auf jeder Stärke; m = 0 (Duelle allein) oder m = f̂(x) (Hybrid, Merkmale als Prior).

λ_u des Hybrids wird je Gruppe über die Laplace-Näherung der Evidenz ln p(D | λ) ≈ ln post(ŝ) + (n/2) ln λ − ½ ln det H auf einem Gitter gewählt (wie beim Empirical Bayes
in `rangintervalle-demo`; kein Orakelwert)."""

import numpy as np

import ltr_constants as C
from ltr_models import sigmoid


def bt_prior(W: np.ndarray, m: np.ndarray, lam: float, iters: int = 10) -> tuple:
    """MAP der Stärken einer Gruppe mit Prior-Mittel m und Präzision λ: (ŝ, ln Evidenz). Newton, gedämpft nicht nötig (starker Prior um m)."""
    n = len(W)
    G = W + W.T
    s = m.astype(float).copy()
    for _ in range(iters):
        P = sigmoid(s[:, None] - s[None, :])
        grad = (W - G * P).sum(axis=1) - lam * (s - m)
        h = G * P * (1.0 - P)
        np.fill_diagonal(h, 0.0)
        H = lam * np.eye(n) + np.diag(h.sum(axis=1)) - h
        s = s + np.linalg.solve(H, grad)
    P = sigmoid(s[:, None] - s[None, :])
    h = G * P * (1.0 - P)
    np.fill_diagonal(h, 0.0)
    H = lam * np.eye(n) + np.diag(h.sum(axis=1)) - h
    logpost = -(W * np.logaddexp(0.0, -(s[:, None] - s[None, :]))).sum() - 0.5 * lam * np.sum((s - m) ** 2)
    return s, float(logpost + 0.5 * n * np.log(lam) - 0.5 * np.linalg.slogdet(H)[1])


def duels_only(W: np.ndarray, lam: float = 0.1) -> np.ndarray:
    """Bradley-Terry der Gruppe allein (Ridge, Prior-Mittel 0)."""
    return bt_prior(W, np.zeros(len(W)), lam)[0]


def hybrid(W: np.ndarray, prior_mean: np.ndarray, grid=C.LAM_GRID) -> tuple:
    """(ŝ, gewähltes λ_u): die Evidenz wählt aus dem Gitter."""
    best = None
    for lam in grid:
        s, ev = bt_prior(W, prior_mean, lam)
        if best is None or ev > best[0]:
            best = (ev, s, lam)
    return best[1], best[2]
