"""Bewertungsfunktionen f(x) (linear oder ein Netz mit einer verdeckten Schicht) und fünf Verluste für Ranking aus Gruppen; Training mit Adam in fester Schrittzahl.

Alle Gruppen haben dieselbe Größe m, daher laufen die Verluste über (Q, m)-Matrizen. Daten je Gruppe: Merkmale X (Q, m, p), Siegmatrix W (Q, m, m) mit W[q, i, j] = Siege von i gegen j,
Siege je Spieler `pts`. Relevanzstufen 0 bis 4 aus dem Rang der Siege in der Gruppe (Stufe 4: beste Fünftel; Gleichstände nach Index).

Verluste (als Ableitung nach f, geteilt durch die Zahl der Spieler N):
  punktweise     (1/N) Σ (f − pc)²                      pc = Siege minus Gruppenmittel, geteilt durch die Streuung aller
  paarweise      −Σ_{i≠j} [W_ij ln σ(f_i − f_j)]        logistisch auf den Duellen (Bradley-Terry mit Merkmalen)
  stufen         −Σ_{Stufe i > Stufe j} ln σ(f_i − f_j) RankNet
  listenweise    −Σ_i t_i ln softmax(f)_i               ListNet; t = softmax(pc)
  ndcg           RankNet mit dem Gewicht |Δ Gewinn| · |Δ Abschlag| des aktuellen Rangs (LambdaRank-artig), Faktor 3"""

import numpy as np

import ltr_constants as C


def sigmoid(x):
    return 0.5 * (1.0 + np.tanh(0.5 * np.asarray(x, dtype=float)))


def grades_from(values: np.ndarray) -> np.ndarray:
    """Relevanzstufen 0 bis 4 je Gruppe (letzte Achse) aus den Werten; der beste Fünftel bekommt 4."""
    m = values.shape[-1]
    rank = np.argsort(np.argsort(-values, axis=-1, kind="stable"), axis=-1, kind="stable")
    return np.minimum(4 - (rank * 5 // m), 4).astype(int)


def ndcg_at(scores: np.ndarray, gains: np.ndarray, k: int = C.NDCG_K) -> float:
    """NDCG@k einer Gruppe: DCG der nach `scores` geordneten Gewinne durch das DCG der idealen Ordnung."""
    order = np.argsort(-scores, kind="stable")[:k]
    disc = 1.0 / np.log2(np.arange(2, len(order) + 2))
    dcg = float(np.sum(gains[order] * disc))
    ideal = np.sort(gains)[::-1][:k]
    idcg = float(np.sum(ideal * disc[: len(ideal)]))
    return dcg / idcg if idcg > 0 else 1.0


def pair_accuracy(est: np.ndarray, truth: np.ndarray) -> float:
    """Anteil richtig geordneter Spielerpaare einer Gruppe; ein Gleichstand zählt halb."""
    iu = np.triu_indices(len(truth), 1)
    a, b = np.sign(est[iu[0]] - est[iu[1]]), np.sign(truth[iu[0]] - truth[iu[1]])
    return float(np.where(a == 0, 0.5, (a == b).astype(float)).mean())


def evaluate_scores(scores: np.ndarray, s_true: np.ndarray) -> tuple:
    """(NDCG@5, Paartreffer) gemittelt über die Gruppen; Gewinne 2^Stufe − 1 mit den Stufen der wahren Stärke."""
    gains = 2.0 ** grades_from(s_true) - 1.0
    nd = [ndcg_at(scores[q], gains[q]) for q in range(len(scores))]
    pa = [pair_accuracy(scores[q], s_true[q]) for q in range(len(scores))]
    return float(np.mean(nd)), float(np.mean(pa))


# --- Bewertungsfunktion ----------------------------------------------------------------------------------------------------------------------

class Scorer:
    """f(x) = x·β (hidden = 0) oder w₂·tanh(W₁ x + b₁)."""

    def __init__(self, p: int, hidden: int, init: np.ndarray | None = None):
        self.p, self.hidden = p, hidden
        if hidden:
            z = np.zeros(p * hidden + hidden) if init is None else init[: p * hidden + hidden]
            self.W1 = 0.5 * z[: p * hidden].reshape(p, hidden).copy()
            self.b1 = np.zeros(hidden)
            self.w2 = 0.5 * z[p * hidden: p * hidden + hidden].copy()
            self.params = [self.W1, self.b1, self.w2]
        else:
            self.beta = np.zeros(p)
            self.params = [self.beta]

    def score(self, X: np.ndarray) -> np.ndarray:
        """X (..., p) -> f (...)."""
        flat = X.reshape(-1, self.p)
        if self.hidden:
            self._H = np.tanh(flat @ self.W1 + self.b1)
            f = self._H @ self.w2
        else:
            f = flat @ self.beta
        return f.reshape(X.shape[:-1])

    def backward(self, X: np.ndarray, g: np.ndarray) -> list:
        flat, gf = X.reshape(-1, self.p), g.reshape(-1)
        if self.hidden:
            dH = np.outer(gf, self.w2) * (1.0 - self._H ** 2)
            return [flat.T @ dH, dH.sum(axis=0), self._H.T @ gf]
        return [flat.T @ gf]


def loss_grad(kind: str, f: np.ndarray, W: np.ndarray, pts: np.ndarray, grades: np.ndarray, scale: float) -> np.ndarray:
    """Ableitung des Verlusts nach f (Q, m), geteilt durch die Zahl der Spieler."""
    Q, m = f.shape
    n = Q * m
    pc = (pts - pts.mean(axis=1, keepdims=True)) / scale
    if kind == C.LOSSES[0]:
        return 2.0 * (f - pc) / n
    P = sigmoid(f[:, :, None] - f[:, None, :])
    if kind == C.LOSSES[1]:
        G = W + np.swapaxes(W, 1, 2)
        return -(W - G * P).sum(axis=2) / n
    better = (grades[:, :, None] > grades[:, None, :]).astype(float)
    if kind == C.LOSSES[2]:
        L = better * (1.0 - P)
        return (-L.sum(axis=2) + L.sum(axis=1)) / n
    if kind == C.LOSSES[3]:
        e = np.exp(f - f.max(axis=1, keepdims=True))
        pr = e / e.sum(axis=1, keepdims=True)
        t = np.exp(pc - pc.max(axis=1, keepdims=True))
        t = t / t.sum(axis=1, keepdims=True)
        return (pr - t) / n
    if kind == C.LOSSES[4]:
        gains = 2.0 ** grades - 1.0
        rank = np.argsort(np.argsort(-f, axis=1, kind="stable"), axis=1, kind="stable")
        disc = 1.0 / np.log2(rank + 2.0)
        dn = np.abs(gains[:, :, None] - gains[:, None, :]) * np.abs(disc[:, :, None] - disc[:, None, :])
        L = better * (1.0 - P) * dn
        return 3.0 * (-L.sum(axis=2) + L.sum(axis=1)) / n
    raise ValueError(f"unbekannter Verlust: {kind}")


def total_loss(kind: str, f: np.ndarray, W: np.ndarray, pts: np.ndarray, grades: np.ndarray, scale: float) -> float:
    """Der Wert des Verlusts (zum Prüfen der Ableitungen gegen finite Differenzen; für ndcg der Wert mit festem Gewicht)."""
    Q, m = f.shape
    n = Q * m
    pc = (pts - pts.mean(axis=1, keepdims=True)) / scale
    if kind == C.LOSSES[0]:
        return float(np.sum((f - pc) ** 2) / n)
    d = f[:, :, None] - f[:, None, :]
    logsig = -np.logaddexp(0.0, -d)
    if kind == C.LOSSES[1]:
        return float(-(W * logsig).sum() / n)
    better = (grades[:, :, None] > grades[:, None, :]).astype(float)
    if kind == C.LOSSES[2]:
        return float(-(better * logsig).sum() / n)
    if kind == C.LOSSES[3]:
        lse = np.log(np.exp(f - f.max(axis=1, keepdims=True)).sum(axis=1, keepdims=True)) + f.max(axis=1, keepdims=True)
        t = np.exp(pc - pc.max(axis=1, keepdims=True))
        t = t / t.sum(axis=1, keepdims=True)
        return float(-(t * (f - lse)).sum() / n)
    gains = 2.0 ** grades - 1.0
    rank = np.argsort(np.argsort(-f, axis=1, kind="stable"), axis=1, kind="stable")
    disc = 1.0 / np.log2(rank + 2.0)
    dn = np.abs(gains[:, :, None] - gains[:, None, :]) * np.abs(disc[:, :, None] - disc[:, None, :])
    return float(-3.0 * (better * dn * logsig).sum() / n)


def fit(X: np.ndarray, W: np.ndarray, kind: str, hidden: int = 0, l2: float = C.DEFAULT_L2, init: np.ndarray | None = None,
        iters: int = C.ADAM_ITERS, lr: float = C.ADAM_LR) -> Scorer:
    """Adam mit fester Schrittzahl auf dem gewählten Verlust (plus l2 · Parameter). Keine Konvergenzprüfung: die Zahl der Schritte ist Teil des Verfahrens."""
    pts = W.sum(axis=2)
    grades = grades_from(pts)
    scale = float(np.sqrt(np.mean((pts - pts.mean(axis=1, keepdims=True)) ** 2))) + 1e-9
    net = Scorer(X.shape[-1], hidden, init)
    mom = [np.zeros_like(p) for p in net.params]
    var = [np.zeros_like(p) for p in net.params]
    for t in range(1, iters + 1):
        f = net.score(X)
        grads = net.backward(X, loss_grad(kind, f, W, pts, grades, scale))
        for k, (par, gr) in enumerate(zip(net.params, grads)):
            gr = gr + l2 * par
            mom[k] = 0.9 * mom[k] + 0.1 * gr
            var[k] = 0.999 * var[k] + 0.001 * gr ** 2
            par -= lr * (mom[k] / (1 - 0.9 ** t)) / (np.sqrt(var[k] / (1 - 0.999 ** t)) + 1e-8)
    return net
