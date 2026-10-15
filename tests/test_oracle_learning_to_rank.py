"""Orakel: Anpassung gegen scipy, NDCG gegen Aufzählen, Eigenschaften der Verfahren über feste Läufe."""

import math

import numpy as np
import pytest

import ltr_constants as C
import ltr_evaluation as ev
import ltr_hybrid as H
import ltr_models as M
import ltr_scenario as SC


def _problem(seed=0, Q=30, m=12, p=3):
    rng = np.random.default_rng(seed)
    X = rng.normal(size=(Q, m, p))
    w = np.array([1.0, -0.7, 0.3])
    s = X @ w + 0.5 * rng.normal(size=(Q, m))
    W = np.zeros((Q, m, m))
    for q in range(Q):
        for i in range(m):
            for j in range(i + 1, m):
                k = rng.binomial(2, 1 / (1 + math.exp(-(s[q, i] - s[q, j]))))
                W[q, i, j], W[q, j, i] = k, 2 - k
    return X, W


def test_pairwise_fit_equals_the_scipy_minimiser():
    optimize = pytest.importorskip("scipy.optimize")
    X, W = _problem()
    Q, m, p = X.shape
    l2 = 0.01
    n = Q * m

    def obj(beta):
        f = X @ beta
        d = f[:, :, None] - f[:, None, :]
        return float(-(W * -np.logaddexp(0, -d)).sum() / n + 0.5 * l2 * beta @ beta)

    res = optimize.minimize(obj, np.zeros(p), method="BFGS", options={"gtol": 1e-10})
    ours = M.fit(X, W, C.LOSSES[1], 0, l2, iters=3000, lr=0.02)
    assert ours.beta == pytest.approx(res.x, abs=2e-3)


def test_ndcg_equals_an_enumeration_with_math_log2():
    rng = np.random.default_rng(3)
    for _ in range(20):
        scores = rng.normal(size=9)
        gains = rng.integers(0, 5, size=9).astype(float) ** 2
        order = sorted(range(9), key=lambda i: -scores[i])
        dcg = sum(gains[order[r]] / math.log2(r + 2) for r in range(5))
        ideal = sorted(gains, reverse=True)
        idcg = sum(ideal[r] / math.log2(r + 2) for r in range(5))
        assert M.ndcg_at(scores, gains, 5) == pytest.approx(dcg / idcg if idcg else 1.0)


def test_pair_accuracy_equals_an_enumeration():
    rng = np.random.default_rng(4)
    est, truth = rng.normal(size=10), rng.normal(size=10)
    hits = sum(1.0 if (est[i] > est[j]) == (truth[i] > truth[j]) else 0.0 for i in range(10) for j in range(i + 1, 10))
    assert M.pair_accuracy(est, truth) == pytest.approx(hits / 45)


def test_with_nothing_unexplained_and_many_groups_the_order_is_almost_perfect():
    r = ev.run(C.DEFAULT_P, C.TRUTHS[0], 0.0, 80, 1, C.LOSSES[1], 0, C.DEFAULT_L2, 7)
    assert r["test"][1] > 0.96 and r["ceiling"][1] == 1.0 and r["test"][0] > 0.97


def test_with_much_unexplained_the_order_stays_near_chance_ceiling():
    rs = [ev.run(C.DEFAULT_P, C.TRUTHS[0], 6.0, 40, 1, C.LOSSES[1], 0, C.DEFAULT_L2, sd) for sd in range(6)]
    m = float(np.mean([r["test"][1] for r in rs]))
    assert 0.5 < m < 0.62                                                            # kaum besser als Zufall
    assert float(np.mean([r["ceiling"][1] for r in rs])) < 0.62


def test_test_accuracy_never_exceeds_the_ceiling_by_much_and_random_is_near_one_half():
    for sd in range(5):
        r = ev.run(C.DEFAULT_P, C.TRUTHS[0], 0.7, 80, 1, C.LOSSES[1], 0, C.DEFAULT_L2, sd)
        assert r["test"][1] <= r["ceiling"][1] + 0.03
        assert 0.45 < r["random"][1] < 0.55


def test_training_gap_appears_with_many_features_and_few_groups():
    rs = [ev.run(30, C.TRUTHS[0], 0.5, 5, 4, C.LOSSES[1], 0, C.DEFAULT_L2, sd) for sd in range(8)]
    assert float(np.mean([r["train"][1] - r["test"][1] for r in rs])) > 0.05


def test_hybrid_equals_duels_without_a_prior_signal_and_features_without_duels():
    te = SC.make_groups(SC.make_truth(8, C.TRUTHS[0], 0.7, 1), 5, 1, 1, 1000, 1.0)
    f = te.f_true
    only_features = SC.make_groups(SC.make_truth(8, C.TRUTHS[0], 0.7, 1), 5, 1, 1, 1000, 0.0)
    for g in range(5):
        hyb, _ = H.hybrid(only_features.W[g], f[g])
        assert hyb == pytest.approx(f[g])                                            # keine Duelle: der Prior bleibt
    many = SC.make_groups(SC.make_truth(8, C.TRUTHS[0], 0.7, 1), 3, 60, 1, 1000, 1.0)
    for g in range(3):
        hyb, _ = H.hybrid(many.W[g], np.zeros(12))
        assert M.pair_accuracy(hyb, many.s[g]) > 0.9                                 # viele Duelle: die Duelle tragen


def test_the_net_overfits_few_groups_with_a_nonlinear_truth():
    r = ev.run(C.DEFAULT_P, C.TRUTHS[1], 0.5, 20, 4, C.LOSSES[1], C.HIDDEN, C.DEFAULT_L2, 3)
    assert r["train"][1] > r["test"][1]
