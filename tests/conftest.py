"""Gemeinsame Fixtures: Pfad zum Demo-Ordner und von Hand gebaute Siegmatrizen (DEMO-PLAYBOOK §1b)."""

import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))


def counts(n, wins):
    """Siegmatrix aus {(i, j): Siege von i gegen j}; die Partienmatrix ist die Summe beider Richtungen."""
    W = np.zeros((n, n))
    for (i, j), w in wins.items():
        W[i, j] = w
    return W, W + W.T
