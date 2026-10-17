"""Eigenschaftstests (metamorph) zu Learning to Rank (ORACLE-PLAYBOOK §5.2): Aussagen über NDCG, die Verluste und ihre Ableitungen, die für das Verfahren gelten müssen, ohne dass das
richtige Ergebnis bekannt ist. Je Eigenschaft 20 bis 40 Zufallsinstanzen mit festem Seed plus Randfälle.

Voraussetzungen, die in den Tests stehen:
- Relevanzstufen (`grades_from`) brechen Gleichstände der Siege nach dem Index. Beim Umsortieren der Spieler einer Liste werden die Stufen deshalb mitgeführt (nicht neu berechnet);
  die Aussagen zu NDCG und Verlusten gelten für beliebige Stufen, und die Wertungen f der Instanzen sind stetig gezogen (keine Gleichstände, die der Index entscheiden würde).
- Verschiebung aller Scores einer Liste um dieselbe Konstante ändert vier der fünf Verluste nicht (paarweise, Stufen, listenweise, NDCG-gewichtet: sie hängen nur von Differenzen
  f_i − f_j beziehungsweise von der Softmax ab). Der punktweise Verlust (f gegen die zentrierten Siege) hängt vom absoluten Niveau ab und ist ausgenommen."""

import math

import numpy as np
import pytest

import ltr_constants as C
import ltr_models as M

SHIFT_INVARIANT = C.LOSSES[1:]
LN2 = math.log(2.0)


def _lists(rng, Q, m, p=3):
    """Zufällige Gruppen: Merkmale, Wertung f, Siegmatrix W (je Paar 0 bis 4 Partien), Siege, Stufen, Streuung."""
    X = rng.normal(size=(Q, m, p))
    f = rng.normal(size=(Q, m))
    W = np.zeros((Q, m, m))
    for q in range(Q):
        for i in range(m):
            for j in range(i + 1, m):
                k = int(rng.integers(0, 5))
                w = int(rng.integers(0, k + 1))
                W[q, i, j], W[q, j, i] = w, k - w
    pts = W.sum(axis=2)
    scale = float(np.sqrt(np.mean((pts - pts.mean(axis=1, keepdims=True)) ** 2))) + 1e-9
    return X, f, W, pts, M.grades_from(pts), scale


def _permute(rng, f, W, pts, grades):
    """Je Liste eine eigene Umnummerierung der Spieler; Stufen werden mitgeführt."""
    Q, m = f.shape
    perms = [rng.permutation(m) for _ in range(Q)]
    f2 = np.stack([f[q][perms[q]] for q in range(Q)])
    W2 = np.stack([W[q][np.ix_(perms[q], perms[q])] for q in range(Q)])
    pts2 = np.stack([pts[q][perms[q]] for q in range(Q)])
    g2 = np.stack([grades[q][perms[q]] for q in range(Q)])
    return perms, f2, W2, pts2, g2


# --- 1. NDCG liegt in [0, 1], ist 1 für die perfekte Ordnung, und die schlechteste Ordnung ist das Minimum ------------------------------------

def test_ndcg_is_in_zero_one_one_for_the_perfect_order_and_minimal_for_the_reversed_order():
    rng = np.random.default_rng(20261017)
    for _ in range(40):
        m = int(rng.integers(2, 15))
        gains = 2.0 ** rng.integers(0, 5, size=m) - 1.0
        k = int(rng.integers(1, m + 3))
        scores = rng.normal(size=m)
        v = M.ndcg_at(scores, gains, k)
        assert 0.0 <= v <= 1.0 + 1e-12
        assert M.ndcg_at(gains + 1e-3 * rng.random(m), gains, k) == pytest.approx(1.0)               # Ordnung nach den Gewinnen ist ideal
        assert M.ndcg_at(np.exp(gains), gains, k) == pytest.approx(1.0)
        full = M.ndcg_at(scores, gains, m)                                                        # volle Länge: Rearrangement-Ungleichung
        assert M.ndcg_at(-gains - 1e-3 * rng.random(m), gains, m) <= full + 1e-12
    assert M.ndcg_at(np.array([1.0, 2.0, 3.0]), np.zeros(3), 5) == 1.0                              # Randfall: kein Gewinn in der Liste
    assert M.ndcg_at(np.array([0.5]), np.array([3.0]), 5) == pytest.approx(1.0)                      # eine Zeile


def test_swapping_a_misordered_pair_never_lowers_ndcg():
    rng = np.random.default_rng(2)
    for _ in range(40):
        m = int(rng.integers(3, 13))
        gains = 2.0 ** rng.integers(0, 5, size=m) - 1.0
        scores = rng.permutation(m).astype(float)                                                  # verschiedene Werte
        k = int(rng.integers(1, m + 1))
        order = np.argsort(-scores)
        # alle Paare von Positionen (a < b) mit gain[order[a]] < gain[order[b]]: Vertauschen verbessert die Ordnung
        for a in range(m):
            for b in range(a + 1, m):
                if gains[order[a]] < gains[order[b]]:
                    s2 = scores.copy()
                    s2[order[a]], s2[order[b]] = scores[order[b]], scores[order[a]]
                    assert M.ndcg_at(s2, gains, k) >= M.ndcg_at(scores, gains, k) - 1e-12
                    break


# --- 2. Umsortieren der Elemente einer Liste ---------------------------------------------------------------------------------------------------

def test_reordering_the_items_of_a_list_changes_neither_ndcg_nor_pair_accuracy():
    rng = np.random.default_rng(3)
    for _ in range(40):
        m = int(rng.integers(2, 15))
        scores, s_true = rng.normal(size=m), rng.normal(size=m)
        gains = 2.0 ** M.grades_from(s_true) - 1.0
        p = rng.permutation(m)
        for k in (1, 5, m):
            assert M.ndcg_at(scores[p], gains[p], k) == pytest.approx(M.ndcg_at(scores, gains, k))
        assert M.pair_accuracy(scores[p], s_true[p]) == pytest.approx(M.pair_accuracy(scores, s_true))
    S, T = rng.normal(size=(6, 12)), rng.normal(size=(6, 12))
    perms = [rng.permutation(12) for _ in range(6)]
    nd, pa = M.evaluate_scores(S, T)
    nd2, pa2 = M.evaluate_scores(np.stack([S[q][perms[q]] for q in range(6)]), np.stack([T[q][perms[q]] for q in range(6)]))
    assert nd2 == pytest.approx(nd) and pa2 == pytest.approx(pa)


@pytest.mark.parametrize("kind", C.LOSSES)
def test_reordering_the_items_of_every_list_changes_neither_the_loss_nor_the_gradient(kind):
    rng = np.random.default_rng(4)
    for _ in range(25):
        Q, m = int(rng.integers(1, 5)), int(rng.integers(2, 13))
        _, f, W, pts, grades, scale = _lists(rng, Q, m)
        perms, f2, W2, pts2, g2 = _permute(rng, f, W, pts, grades)
        assert M.total_loss(kind, f2, W2, pts2, g2, scale) == pytest.approx(M.total_loss(kind, f, W, pts, grades, scale), rel=1e-9, abs=1e-12)
        g, gp = M.loss_grad(kind, f, W, pts, grades, scale), M.loss_grad(kind, f2, W2, pts2, g2, scale)
        assert gp == pytest.approx(np.stack([g[q][perms[q]] for q in range(Q)]), abs=1e-12)
        order = rng.permutation(Q)                                                                  # auch die Reihenfolge der Listen ist gleichgültig
        assert M.total_loss(kind, f[order], W[order], pts[order], grades[order], scale) == pytest.approx(M.total_loss(kind, f, W, pts, grades, scale), rel=1e-9, abs=1e-12)


# --- 3. Positive Skalierung der wahren Relevanz ändert die Ordnung nicht --------------------------------------------------------------------

def test_positive_scaling_and_shifting_of_the_true_relevance_changes_no_grade_and_no_measure():
    rng = np.random.default_rng(5)
    for _ in range(40):
        Q, m = int(rng.integers(1, 5)), int(rng.integers(2, 16))
        s_true = rng.normal(size=(Q, m))
        c, b = float(rng.uniform(0.01, 100.0)), float(rng.normal(scale=10.0))
        g = M.grades_from(s_true)
        assert np.array_equal(M.grades_from(c * s_true + b), g)
        assert g.min() >= 0 and g.max() <= 4
        scores = rng.normal(size=(Q, m))
        assert M.evaluate_scores(scores, c * s_true + b) == pytest.approx(M.evaluate_scores(scores, s_true))
        assert M.evaluate_scores(c * scores + b, s_true) == pytest.approx(M.evaluate_scores(scores, s_true))              # auch positive Skalierung der Scores
        # höherer Wert -> nie eine niedrigere Stufe
        for q in range(Q):
            o = np.argsort(s_true[q])
            assert (np.diff(g[q][o]) >= 0).all()
        gains = 2.0 ** g[0] - 1.0
        assert M.ndcg_at(scores[0], c * gains, 5) == pytest.approx(M.ndcg_at(scores[0], gains, 5))                          # NDCG ist ein Verhältnis
    assert M.grades_from(np.arange(5.0)[None, :]).tolist() == [[0, 1, 2, 3, 4]]                                           # Randfall: fünf Spieler, eine Stufe je Platz


# --- 4. Gleiche Scores: Paarverlust = log 2 je Partie, Liste = log m ----------------------------------------------------------------------------

def test_with_equal_scores_each_game_costs_log_two_and_a_list_costs_log_m():
    rng = np.random.default_rng(6)
    for _ in range(30):
        Q, m = int(rng.integers(1, 5)), int(rng.integers(2, 13))
        _, _, W, pts, grades, scale = _lists(rng, Q, m)
        f = np.full((Q, m), float(rng.normal()))
        n = Q * m
        assert M.total_loss(C.LOSSES[1], f, W, pts, grades, scale) == pytest.approx(LN2 * W.sum() / n)
        better = (grades[:, :, None] > grades[:, None, :]).sum()
        assert M.total_loss(C.LOSSES[2], f, W, pts, grades, scale) == pytest.approx(LN2 * better / n)
        assert M.total_loss(C.LOSSES[3], f, W, pts, grades, scale) == pytest.approx(math.log(m) / m)
        # Ableitung des Paarverlusts bei gleichen Scores: −(Σ_j W_ij − G_ij/2) / n
        G = W + np.swapaxes(W, 1, 2)
        assert M.loss_grad(C.LOSSES[1], f, W, pts, grades, scale) == pytest.approx(-(W.sum(axis=2) - G.sum(axis=2) / 2.0) / n)
    # Randfall: ein Sieg, zwei Spieler, gleiche Scores: log 2 je Partie (Faktor n = 2)
    W1 = np.array([[[0.0, 1.0], [0.0, 0.0]]])
    assert 2 * M.total_loss(C.LOSSES[1], np.zeros((1, 2)), W1, W1.sum(axis=2), M.grades_from(W1.sum(axis=2)), 1.0) == pytest.approx(LN2)


# --- 5. Verschiebung aller Scores ----------------------------------------------------------------------------------------------------------------

@pytest.mark.parametrize("kind", SHIFT_INVARIANT)
def test_shifting_all_scores_of_a_list_by_a_constant_changes_neither_the_loss_nor_the_gradient(kind):
    rng = np.random.default_rng(7)
    for _ in range(30):
        Q, m = int(rng.integers(1, 5)), int(rng.integers(2, 13))
        _, f, W, pts, grades, scale = _lists(rng, Q, m)
        shift = rng.normal(scale=20.0, size=(Q, 1))                                                  # je Liste eine eigene Konstante
        assert M.total_loss(kind, f + shift, W, pts, grades, scale) == pytest.approx(M.total_loss(kind, f, W, pts, grades, scale), rel=1e-9, abs=1e-12)
        g = M.loss_grad(kind, f, W, pts, grades, scale)
        assert M.loss_grad(kind, f + shift, W, pts, grades, scale) == pytest.approx(g, abs=1e-10)
        assert g.sum(axis=1) == pytest.approx(np.zeros(Q), abs=1e-12)                              # Folge: die Ableitung summiert je Liste zu 0


@pytest.mark.parametrize("kind", [C.LOSSES[1], C.LOSSES[3]])
def test_a_linear_fit_ignores_a_common_shift_of_all_features(kind):
    rng = np.random.default_rng(8)
    for _ in range(4):
        X, _, W, _, _, _ = _lists(rng, 8, 12, 3)
        shift = rng.normal(scale=5.0, size=3)
        a = M.fit(X, W, kind, hidden=0, iters=20)
        b = M.fit(X + shift, W, kind, hidden=0, iters=20)
        assert b.beta == pytest.approx(a.beta, abs=1e-6)
        order = rng.permutation(8)
        assert M.fit(X[order], W[order], kind, hidden=0, iters=20).beta == pytest.approx(a.beta, abs=1e-6)             # Reihenfolge der Listen


# --- 6. Ableitungen gegen finite Differenzen -----------------------------------------------------------------------------------------------------

@pytest.mark.parametrize("kind", C.LOSSES)
def test_every_loss_gradient_matches_central_differences_on_random_instances(kind):
    rng = np.random.default_rng(9)
    h = 1e-5
    for _ in range(25):
        Q, m = int(rng.integers(1, 5)), int(rng.integers(2, 13))
        _, f, W, pts, grades, scale = _lists(rng, Q, m)
        g = M.loss_grad(kind, f, W, pts, grades, scale)
        assert g.shape == f.shape and np.isfinite(g).all()
        for _ in range(6):
            q, i = int(rng.integers(0, Q)), int(rng.integers(0, m))
            up, dn = f.copy(), f.copy()
            up[q, i] += h
            dn[q, i] -= h
            fd = (M.total_loss(kind, up, W, pts, grades, scale) - M.total_loss(kind, dn, W, pts, grades, scale)) / (2 * h)
            assert g[q, i] == pytest.approx(fd, abs=2e-7), (kind, Q, m, q, i)


@pytest.mark.parametrize("kind", C.LOSSES)
@pytest.mark.parametrize("hidden", [0, 3])
def test_the_parameter_gradient_through_the_scorer_matches_central_differences(kind, hidden):
    rng = np.random.default_rng(10)
    h = 1e-6
    for _ in range(5):
        Q, m, p = int(rng.integers(1, 4)), int(rng.integers(3, 10)), 3
        X, _, W, pts, grades, scale = _lists(rng, Q, m, p)
        net = M.Scorer(p, hidden, rng.normal(size=p * 3 + 3) if hidden else None)
        if not hidden:
            net.beta[:] = rng.normal(size=p)
        grads = net.backward(X, M.loss_grad(kind, net.score(X), W, pts, grades, scale))
        for par, gr in zip(net.params, grads):
            flat, gflat = par.reshape(-1), gr.reshape(-1)
            for idx in rng.choice(len(flat), size=min(len(flat), 4), replace=False):
                old = flat[idx]
                flat[idx] = old + h
                up = M.total_loss(kind, net.score(X), W, pts, grades, scale)
                flat[idx] = old - h
                dn = M.total_loss(kind, net.score(X), W, pts, grades, scale)
                flat[idx] = old
                assert gflat[idx] == pytest.approx((up - dn) / (2 * h), abs=5e-7), (kind, hidden, idx)


# --- 7. Listenverlust: Kreuzentropie ist beim Zielwert am kleinsten ----------------------------------------------------------------------------

def test_the_listwise_loss_is_smallest_when_the_scores_equal_the_target_scores():
    """ListNet: Kreuzentropie der Top-eins-Wahrscheinlichkeiten; sie ist minimal, wenn die Wahrscheinlichkeiten übereinstimmen (Gibbs), hier bei f = zentrierte Siege / Streuung
    (und jeder Verschiebung davon)."""
    rng = np.random.default_rng(11)
    for _ in range(30):
        Q, m = int(rng.integers(1, 5)), int(rng.integers(2, 13))
        _, _, W, pts, grades, scale = _lists(rng, Q, m)
        pc = (pts - pts.mean(axis=1, keepdims=True)) / scale
        best = M.total_loss(C.LOSSES[3], pc, W, pts, grades, scale)
        assert M.loss_grad(C.LOSSES[3], pc, W, pts, grades, scale) == pytest.approx(np.zeros((Q, m)), abs=1e-12)
        assert M.total_loss(C.LOSSES[3], pc + rng.normal(scale=7.0, size=(Q, 1)), W, pts, grades, scale) == pytest.approx(best, abs=1e-12)
        for _ in range(5):
            assert M.total_loss(C.LOSSES[3], pc + rng.normal(scale=0.5, size=(Q, m)), W, pts, grades, scale) >= best - 1e-12
