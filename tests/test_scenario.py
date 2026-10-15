"""Simulation: Wahrheit, Gruppen, Duelle, Hybrid an Handwerten."""

import numpy as np
import pytest

import ltr_constants as C
import ltr_hybrid as H
import ltr_scenario as SC
import turnier_generator as T


def test_truth_vectors_are_unit_and_deterministic():
    t = SC.make_truth(8, C.TRUTHS[0], 0.7, 5)
    assert np.linalg.norm(t.w) == pytest.approx(1.0) and np.linalg.norm(t.v) == pytest.approx(1.0) and not t.nonlinear and t.sigma_u == 0.7
    u = SC.make_truth(8, C.TRUTHS[0], 0.7, 5)
    assert np.array_equal(t.w, u.w) and not np.array_equal(t.w, SC.make_truth(8, C.TRUTHS[0], 0.7, 6).w)
    assert SC.make_truth(8, C.TRUTHS[1], 0.7, 5).nonlinear


def test_normals_are_box_muller_by_hand():
    import math
    for count in (2, 3):
        u = T.SplitMix64(11)
        half = (count + 1) // 2
        u1, u2 = u.uniform(half), u.uniform(half)
        r = np.sqrt(-2.0 * np.log(u1))
        expected = np.concatenate([r * np.cos(2 * math.pi * u2), r * np.sin(2 * math.pi * u2)])[:count]
        assert SC.normals(T.SplitMix64(11), count) == pytest.approx(expected)
    z = SC.normals(T.SplitMix64(5), 20001)
    assert abs(z.mean()) < 0.03 and z.std() == pytest.approx(1.0, abs=0.03)


def test_linear_and_nonlinear_functions_by_hand():
    lin = SC.Truth(np.array([1.0, 0.0]), np.array([0.0, 1.0]), False, 0.0)
    X = np.array([[2.0, 3.0], [0.0, 0.0]])
    assert lin.f(X).tolist() == [2.0, 0.0]
    nl = SC.Truth(np.array([1.0, 0.0]), np.array([0.0, 1.0]), True, 0.0)
    expected = (0.8 * np.tanh(2 * 2.0) + 0.6 * np.sin(1.5 * 3.0) + 0.4 * 2.0 * 3.0) / 0.9
    assert nl.f(X)[0] == pytest.approx(expected) and nl.f(X)[1] == 0.0


def test_groups_are_deterministic_and_prefix_stable():
    t = SC.make_truth(5, C.TRUTHS[0], 0.7, 3)
    a = SC.make_groups(t, 6, 2, 3)
    b = SC.make_groups(t, 10, 2, 3)
    assert np.array_equal(a.X, b.X[:6]) and np.array_equal(a.W, b.W[:6])             # die ersten Q Gruppen bleiben gleich
    assert not np.array_equal(a.X, SC.make_groups(t, 6, 2, 4).X)
    other = SC.make_groups(t, 3, 2, 3, offset=1000)
    assert not np.array_equal(other.X, a.X[:3])


def test_duels_k_games_per_pair_and_the_win_matrix_is_consistent():
    t = SC.make_truth(4, C.TRUTHS[0], 0.5, 1)
    g = SC.make_groups(t, 5, 3, 1)
    G = g.W + np.swapaxes(g.W, 1, 2)
    iu = np.triu_indices(12, 1)
    assert g.X.shape == (5, 12, 4) and g.s.shape == (5, 12) and g.W.shape == (5, 12, 12)
    assert (G[:, iu[0], iu[1]] == 3).all() and (np.einsum("qii->qi", g.W) == 0).all()


def test_test_groups_play_the_requested_share_once():
    t = SC.make_truth(4, C.TRUTHS[0], 0.5, 1)
    none = SC.make_groups(t, 6, 1, 1, offset=1000, q=0.0)
    assert none.W.sum() == 0
    full = SC.make_groups(t, 6, 1, 1, offset=1000, q=1.0)
    assert (full.W + np.swapaxes(full.W, 1, 2))[:, 0, 1].tolist() == [1.0] * 6
    half = SC.make_groups(t, 60, 1, 1, offset=1000, q=0.5)
    share = (half.W + np.swapaxes(half.W, 1, 2))[:, np.triu_indices(12, 1)[0], np.triu_indices(12, 1)[1]].mean()
    assert share == pytest.approx(0.5, abs=0.04)
    cold = SC.make_groups(t, 3, 0, 1, q=1.0)
    assert cold.W.sum() == 0                                                         # ohne Duelle k = 0


def test_unexplained_part_has_the_requested_spread_and_stronger_players_win_more():
    t = SC.make_truth(6, C.TRUTHS[0], 1.0, 2)
    g = SC.make_groups(t, 200, 1, 2)
    resid = (g.s - g.f_true).ravel()
    assert resid.std() == pytest.approx(1.0, abs=0.06)
    assert g.f_true.std() == pytest.approx(1.0, abs=0.08)                            # |w| = 1 und x ~ N(0, I): Streuung 1
    big = SC.make_groups(SC.make_truth(6, C.TRUTHS[0], 0.0, 2), 20, 40, 2)
    wins = big.W.sum(axis=2)
    assert np.mean([np.corrcoef(wins[q], big.s[q])[0, 1] for q in range(20)]) > 0.95


# --- Hybrid ---------------------------------------------------------------------------------------------------------------------------------

def _duels(s, k, rng):
    n = len(s)
    W = np.zeros((n, n))
    for i in range(n):
        for j in range(i + 1, n):
            w = rng.binomial(k, 1 / (1 + np.exp(-(s[i] - s[j]))))
            W[i, j], W[j, i] = w, k - w
    return W


def test_prior_with_huge_precision_returns_the_prior_mean_and_without_duels_too():
    rng = np.random.default_rng(1)
    s = rng.normal(size=6)
    W = _duels(s, 3, rng)
    m = np.linspace(-1, 1, 6)
    assert H.bt_prior(W, m, 1e6)[0] == pytest.approx(m, abs=1e-3)
    assert H.bt_prior(np.zeros((6, 6)), m, 0.5)[0] == pytest.approx(m)


def test_bt_prior_gradient_is_zero_at_the_solution():
    rng = np.random.default_rng(2)
    s = rng.normal(size=6)
    W = _duels(s, 4, rng)
    m = 0.3 * rng.normal(size=6)
    est, _ = H.bt_prior(W, m, 1.0, iters=40)
    G = W + W.T
    P = 1 / (1 + np.exp(-(est[:, None] - est[None, :])))
    grad = (W - G * P).sum(axis=1) - 1.0 * (est - m)
    assert np.abs(grad).max() < 1e-8


def test_evidence_prefers_a_strong_prior_when_the_prior_mean_is_right():
    rng = np.random.default_rng(3)
    s = rng.normal(size=8)
    W = _duels(s, 2, rng)
    _, lam = H.hybrid(W, s)                                                          # exakter Prior-Mittelwert
    assert lam >= 10.0
    _, lam_bad = H.hybrid(W, -s)                                                     # falscher Prior-Mittelwert
    assert lam_bad < lam


def test_hybrid_beats_duels_with_few_games_and_converges_with_many():
    rng = np.random.default_rng(4)
    gain_few, gain_many = [], []
    for _ in range(40):
        s = rng.normal(size=12)
        prior = s + 0.5 * rng.normal(size=12)
        for k, store in ((1, gain_few), (30, gain_many)):
            W = np.zeros((12, 12))
            keep = np.triu(rng.random((12, 12)) < (0.2 if k == 1 else 1.0), 1)
            for i, j in zip(*np.nonzero(keep)):
                w = rng.binomial(k, 1 / (1 + np.exp(-(s[i] - s[j]))))
                W[i, j], W[j, i] = w, k - w
            d = H.duels_only(W)
            h, _ = H.hybrid(W, prior)
            err = lambda est: float(np.mean((est - est.mean() - (s - s.mean())) ** 2))   # noqa: E731
            store.append(err(d) - err(h))
    assert np.mean(gain_few) > 0.1 and abs(np.mean(gain_many)) < 0.2 and np.mean(gain_few) > np.mean(gain_many)


def test_group_numbers_map_to_streams_and_the_win_frequency_matches_the_logistic():
    t = SC.make_truth(4, C.TRUTHS[0], 0.5, 9)
    two = SC.make_groups(t, 2, 1, 9)
    shifted = SC.make_groups(t, 1, 1, 9, offset=1)
    assert np.array_equal(two.X[1], shifted.X[0]) and np.array_equal(two.W[1], shifted.W[0])
    big = SC.make_groups(t, 40, 200, 9)
    iu = np.triu_indices(12, 1)
    p_true = 1 / (1 + np.exp(-(big.s[:, iu[0]] - big.s[:, iu[1]])))
    freq = big.W[:, iu[0], iu[1]] / 200.0
    assert np.abs(freq - p_true).mean() < 0.04 and np.corrcoef(freq.ravel(), p_true.ravel())[0, 1] > 0.99


def test_evidence_matches_numerical_integration_for_two_players():
    import math
    for a, b, lam, m0 in ((12, 4, 1.0, 0.5), (6, 6, 2.0, 0.0), (20, 2, 0.5, -0.3)):
        W = np.array([[0.0, a], [b, 0.0]])
        mean = np.array([m0, 0.0])
        ours = H.bt_prior(W, mean, lam, iters=40)[1]
        d = np.linspace(-14, 14, 40001)
        var = 2.0 / lam
        logf = a * (-np.logaddexp(0, -d)) + b * (-np.logaddexp(0, d)) - 0.5 * (d - m0) ** 2 / var - 0.5 * math.log(2 * math.pi * var)
        top = logf.max()
        exact = top + math.log(np.trapezoid(np.exp(logf - top), d))
        assert ours == pytest.approx(exact, abs=0.08), (a, b, lam)


def test_evidence_matches_numerical_integration_for_three_players():
    import math
    W = np.array([[0.0, 14.0, 12.0], [4.0, 0.0, 9.0], [2.0, 6.0, 0.0]])
    m = np.array([0.4, 0.0, -0.4])
    lam = 1.5
    ours = H.bt_prior(W, m, lam, iters=40)[1]
    g = np.arange(-4.0, 4.0001, 0.1)
    S0, S1, S2 = np.meshgrid(g, g, g, indexing="ij")
    S = np.stack([S0, S1, S2])
    loglik = np.zeros_like(S0)
    for i in range(3):
        for j in range(3):
            if W[i, j]:
                loglik += W[i, j] * (-np.logaddexp(0, -(S[i] - S[j])))
    logprior = -0.5 * lam * sum((S[i] - m[i]) ** 2 for i in range(3)) + 1.5 * math.log(lam / (2 * math.pi))
    logf = loglik + logprior
    top = logf.max()
    exact = top + math.log(np.exp(logf - top).sum() * 0.1 ** 3)
    assert ours == pytest.approx(exact, abs=0.12)
