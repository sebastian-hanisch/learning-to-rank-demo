"""Stufen, NDCG, Paartreffer, Bewertungsfunktionen, Verluste (Ableitungen gegen finite Differenzen) und das Training an Handwerten."""

import math

import numpy as np
import pytest

import ltr_constants as C
import ltr_models as M

LIN_LOSS = C.LOSSES


def test_grades_follow_the_rank_in_the_group():
    pts = np.array([[0.0, 5.0, 3.0, 1.0, 4.0, 2.0, 6.0, 7.0, 8.0, 9.0]])
    assert M.grades_from(pts)[0].tolist() == [0, 2, 1, 0, 2, 1, 3, 3, 4, 4]          # zehn Spieler: zwei je Stufe, der Beste oben
    assert M.grades_from(np.arange(12.0)[None, :])[0].tolist() == [0, 0, 1, 1, 2, 2, 2, 3, 3, 4, 4, 4]
    assert M.grades_from(np.array([[1.0, 2.0, 3.0], [3.0, 2.0, 1.0]])).tolist() == [[1, 3, 4], [4, 3, 1]]       # Gruppen unabhängig


def test_ndcg_by_hand():
    gains = np.array([3.0, 1.0, 0.0, 7.0])
    # Ordnung nach Bewertung: Spieler 3, 0, 1, 2 -> Gewinne 7, 3, 1, 0
    scores = np.array([2.0, 1.0, 0.0, 3.0])
    assert M.ndcg_at(scores, gains, 4) == pytest.approx(1.0)
    # schlechtere Ordnung: Spieler 0, 3, 2, 1 -> Gewinne 3, 7, 0, 1
    bad = np.array([3.0, 0.0, 1.0, 2.0])
    dcg = 3 / math.log2(2) + 7 / math.log2(3) + 0 / math.log2(4) + 1 / math.log2(5)
    idcg = 7 / math.log2(2) + 3 / math.log2(3) + 1 / math.log2(4) + 0 / math.log2(5)
    assert M.ndcg_at(bad, gains, 4) == pytest.approx(dcg / idcg)
    assert M.ndcg_at(bad, gains, 2) == pytest.approx((3 / math.log2(2) + 7 / math.log2(3)) / (7 / math.log2(2) + 3 / math.log2(3)))
    assert M.ndcg_at(np.zeros(3), np.zeros(3), 2) == 1.0                              # nichts zu gewinnen


def test_pair_accuracy_and_evaluate_scores_by_hand():
    s = np.array([0.0, 1.0, 2.0, 3.0])
    assert M.pair_accuracy(s, s) == 1.0 and M.pair_accuracy(-s, s) == 0.0 and M.pair_accuracy(np.zeros(4), s) == 0.5
    assert M.pair_accuracy(np.array([0.0, 0.0, 1.0, 1.0]), s) == pytest.approx((4 + 2 * 0.5) / 6)
    nd, pa = M.evaluate_scores(np.array([s, -s]), np.array([s, s]))
    assert pa == pytest.approx(0.5) and 0.0 < nd < 1.0
    nd1, pa1 = M.evaluate_scores(np.array([s]), np.array([s]))
    assert nd1 == pytest.approx(1.0) and pa1 == 1.0


def test_scorer_forward_and_backward_match_finite_differences():
    rng = np.random.default_rng(1)
    X = rng.normal(size=(3, 5, 4))
    g = rng.normal(size=(3, 5))
    for hidden in (0, 3):
        net = M.Scorer(4, hidden, rng.normal(size=4 * 3 + 3) if hidden else None)
        if not hidden:
            net.beta[:] = rng.normal(size=4)
        base = net.score(X)
        assert base.shape == (3, 5)
        grads = net.backward(X, g)
        for par, gr in zip(net.params, grads):
            flat, gflat = par.reshape(-1), gr.reshape(-1)
            for idx in range(min(len(flat), 6)):
                old = flat[idx]
                flat[idx] = old + 1e-6
                up = float(np.sum(g * net.score(X)))
                flat[idx] = old - 1e-6
                dn = float(np.sum(g * net.score(X)))
                flat[idx] = old
                assert gflat[idx] == pytest.approx((up - dn) / 2e-6, abs=1e-5), (hidden, idx)


def _toy(seed=0, Q=3, m=6):
    rng = np.random.default_rng(seed)
    f = rng.normal(size=(Q, m))
    W = np.zeros((Q, m, m))
    for q in range(Q):
        for i in range(m):
            for j in range(i + 1, m):
                w = rng.integers(0, 4)
                W[q, i, j], W[q, j, i] = w, 3 - w
    pts = W.sum(axis=2)
    return f, W, pts, M.grades_from(pts), float(np.sqrt(np.mean((pts - pts.mean(axis=1, keepdims=True)) ** 2)))


@pytest.mark.parametrize("kind", C.LOSSES)
def test_every_loss_gradient_matches_finite_differences(kind):
    f, W, pts, grades, scale = _toy()
    g = M.loss_grad(kind, f, W, pts, grades, scale)
    assert g.shape == f.shape
    for q in range(f.shape[0]):
        for i in range(f.shape[1]):
            up, dn = f.copy(), f.copy()
            up[q, i] += 1e-6
            dn[q, i] -= 1e-6
            fd = (M.total_loss(kind, up, W, pts, grades, scale) - M.total_loss(kind, dn, W, pts, grades, scale)) / 2e-6
            assert g[q, i] == pytest.approx(fd, abs=2e-6), (kind, q, i)


def test_unknown_loss_is_rejected():
    f, W, pts, grades, scale = _toy()
    with pytest.raises(ValueError):
        M.loss_grad("unbekannt", f, W, pts, grades, scale)


def test_pointwise_loss_by_hand():
    f = np.array([[1.0, 0.0, -1.0]])
    W = np.zeros((1, 3, 3))
    W[0, 0, 1], W[0, 1, 0] = 2, 0
    W[0, 0, 2], W[0, 2, 0] = 2, 0
    W[0, 1, 2], W[0, 2, 1] = 2, 0                                                      # Siege 4, 2, 0 -> zentriert 2, 0, −2
    pts = W.sum(axis=2)
    scale = 2.0                                                                         # pc = (1, 0, −1) = f
    assert M.total_loss(C.LOSSES[0], f, W, pts, M.grades_from(pts), scale) == pytest.approx(0.0)
    assert M.loss_grad(C.LOSSES[0], f, W, pts, M.grades_from(pts), scale) == pytest.approx(np.zeros((1, 3)))


def test_pairwise_loss_at_zero_by_hand():
    f = np.zeros((1, 2))
    W = np.array([[[0.0, 3.0], [1.0, 0.0]]])
    pts = W.sum(axis=2)
    g = M.loss_grad(C.LOSSES[1], f, W, pts, M.grades_from(pts), 1.0)
    # Verlust −(3 ln σ(f0−f1) + 1 ln σ(f1−f0)) / 2 ; Ableitung nach f0 bei 0: −(3·½ − 1·½)/2 = −½
    assert g[0, 0] == pytest.approx(-0.5) and g[0, 1] == pytest.approx(0.5)
    assert M.total_loss(C.LOSSES[1], f, W, pts, M.grades_from(pts), 1.0) == pytest.approx(4 * math.log(2) / 2)


def test_fit_learns_a_linear_direction_and_is_deterministic():
    rng = np.random.default_rng(5)
    Q, m, p = 60, 12, 3
    X = rng.normal(size=(Q, m, p))
    w = np.array([1.0, -0.5, 0.0])
    s = X @ w
    W = np.zeros((Q, m, m))
    for q in range(Q):
        for i in range(m):
            for j in range(i + 1, m):
                win = rng.random() < 1 / (1 + math.exp(-(s[q, i] - s[q, j])))
                W[q, i, j], W[q, j, i] = (1, 0) if win else (0, 1)
    net = M.fit(X, W, C.LOSSES[1], 0, 0.0)
    assert np.dot(net.beta, w) / (np.linalg.norm(net.beta) * np.linalg.norm(w)) > 0.97
    again = M.fit(X, W, C.LOSSES[1], 0, 0.0)
    assert np.array_equal(net.beta, again.beta)
    stronger = M.fit(X, W, C.LOSSES[1], 0, 1.0)
    assert np.linalg.norm(stronger.beta) < np.linalg.norm(net.beta)                      # Gewichtsstrafe verkleinert die Parameter
    netw = M.fit(X, W, C.LOSSES[1], 4, 0.01, np.random.default_rng(0).normal(size=3 * 4 + 4))
    assert netw.score(X).shape == (Q, m)


def test_evaluate_scores_uses_gains_two_to_the_grade_minus_one():
    # wahre Stärke: Stufen der 12 Spieler [0,0,1,1,2,2,2,3,3,4,4,4] nach aufsteigender Stärke; Bewertung = verkehrte Reihenfolge der besten drei
    s = np.arange(12.0)
    scores = s.copy()
    scores[5], scores[11] = 11.0, 5.0                                                 # ein Spieler der Stufe 2 steht ganz oben, einer der Stufe 4 ist abgerutscht
    nd, pa = M.evaluate_scores(scores[None, :], s[None, :])
    grades = M.grades_from(s[None, :])[0]
    gains = 2.0 ** grades - 1.0
    order = np.argsort(-scores)
    dcg = sum(gains[order[r]] / math.log2(r + 2) for r in range(5))
    idcg = sum(sorted(gains, reverse=True)[r] / math.log2(r + 2) for r in range(5))
    assert nd == pytest.approx(dcg / idcg) and nd < 1.0 and pa < 1.0
    assert gains.max() == 15.0 and gains.min() == 0.0                                 # 2^4 − 1 und 2^0 − 1


def test_ndcg_with_small_total_gain_is_still_a_ratio():
    # nur ein Spieler mit Gewinn 1: IDCG = 1; steht er auf Platz 2, ist NDCG = 1/log2(3)
    assert M.ndcg_at(np.array([1.0, 2.0, 0.0]), np.array([1.0, 0.0, 0.0]), 3) == pytest.approx(1 / math.log2(3))


def test_scorer_without_init_has_zero_weights_and_scores_zero():
    net = M.Scorer(4, 3)
    assert net.score(np.ones((2, 5, 4))).tolist() == np.zeros((2, 5)).tolist()


def _adam_reference(X, W, kind, hidden, l2, iters, lr, init):
    pts = W.sum(axis=2)
    grades = M.grades_from(pts)
    scale = float(np.sqrt(np.mean((pts - pts.mean(axis=1, keepdims=True)) ** 2))) + 1e-9
    net = M.Scorer(X.shape[-1], hidden, init)
    m1 = [np.zeros_like(q) for q in net.params]
    v1 = [np.zeros_like(q) for q in net.params]
    for t in range(1, iters + 1):
        grads = net.backward(X, M.loss_grad(kind, net.score(X), W, pts, grades, scale))
        for i, (par, g) in enumerate(zip(net.params, grads)):
            g = g + l2 * par
            m1[i] = 0.9 * m1[i] + 0.1 * g
            v1[i] = 0.999 * v1[i] + 0.001 * g * g
            mh, vh = m1[i] / (1 - 0.9 ** t), v1[i] / (1 - 0.999 ** t)
            par -= lr * mh / (np.sqrt(vh) + 1e-8)
    return net


@pytest.mark.parametrize("kind", C.LOSSES)
@pytest.mark.parametrize("iters", [1, 3])
def test_fit_follows_the_adam_recursion_for_every_loss(kind, iters):
    rng = np.random.default_rng(7)
    Q, m, p = 6, 12, 3
    X = rng.normal(size=(Q, m, p))
    W = np.zeros((Q, m, m))
    for q in range(Q):
        for i in range(m):
            for j in range(i + 1, m):
                w = rng.integers(0, 3)
                W[q, i, j], W[q, j, i] = w, 2 - w
    for hidden, init in ((0, None), (4, rng.normal(size=3 * 4 + 4))):
        ours = M.fit(X, W, kind, hidden, 0.05, init, iters=iters, lr=0.1)
        ref = _adam_reference(X, W, kind, hidden, 0.05, iters, 0.1, init)
        for a, b in zip(ours.params, ref.params):
            assert a == pytest.approx(b, abs=1e-12), (kind, iters, hidden)
