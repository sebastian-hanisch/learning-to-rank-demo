"""Analyse des gezeigten Laufs, Verdikt-Zweige an handgesetzten Werten und Hilfsfunktionen; die Experimente selbst prüft test_claims.py."""

import numpy as np
import pytest

import ltr_constants as C
import ltr_evaluation as ev
from ltr_scenario import Settings

PUNKT, PAAR, STUFEN, LIST, NDCG = C.LOSSES
LIN, NET = C.MODELS


def test_german_number_formatting_and_helpers():
    assert ev.de(1.5, 1) == "1,5" and ev.de(0.666) == "0,67" and ev.pct(0.256) == "26" and ev.pct(0.256, 1) == "25,6"
    assert ev._mean([1.0, float("nan"), 3.0]) == 2.0 and np.isnan(ev._mean([float("nan")]))
    assert ev.hidden_of(LIN) == 0 and ev.hidden_of(NET) == C.HIDDEN
    assert ev.explained_share(0.0) == 1.0 and ev.explained_share(1.0) == pytest.approx(0.5) and ev.explained_share(0.7) == pytest.approx(1 / 1.49)


def test_analysis_is_consistent_and_deterministic():
    s = Settings(q_groups=12, seed=2)
    a = ev.analyse(s)
    assert a.scores.shape == (C.N_TEST_GROUPS, C.M_GROUP) and a.test.s.shape == a.scores.shape
    assert set(a.metrics) == {"test", "train", "ceiling", "random"}
    assert a.curve[-1][0] == 12 and [c[0] for c in a.curve] == sorted(c[0] for c in a.curve) and a.curve[0][0] == 3
    b = ev.analyse(s)
    assert a.metrics == b.metrics and a.curve == b.curve
    assert a.explained == pytest.approx(1 / (1 + 0.49))


def test_warm_analysis_adds_duels_and_hybrid():
    a = ev.analyse(Settings(q_groups=10, q_test=0.5, seed=1))
    assert set(a.metrics) == {"test", "train", "ceiling", "random", "duels", "hybrid"}
    assert a.metrics["duels"][1] > 0.55 and a.metrics["hybrid"][1] > 0.55


def test_net_and_loss_choice_are_used():
    base = ev.analyse(Settings(q_groups=15, seed=4))
    net = ev.analyse(Settings(q_groups=15, seed=4, model=NET))
    other = ev.analyse(Settings(q_groups=15, seed=4, loss=LIST))
    assert not np.array_equal(base.scores, net.scores) and not np.array_equal(base.scores, other.scores)
    assert net.scores.shape == base.scores.shape


def test_run_cells_are_deterministic_and_share_data_across_losses():
    r1 = ev.run(8, C.TRUTHS[0], 0.7, 10, 1, PAAR, 0, 0.01, 5)
    r2 = ev.run(8, C.TRUTHS[0], 0.7, 10, 1, PAAR, 0, 0.01, 5)
    assert r1 == r2
    r3 = ev.run(8, C.TRUTHS[0], 0.7, 10, 1, PUNKT, 0, 0.01, 5)
    assert r3["ceiling"] == r1["ceiling"] and r3["random"] == r1["random"]               # dieselben Testgruppen und Würfe


# --- Verdikt-Zweige ---------------------------------------------------------------------------------------------------------------------------

def _cell(test=0.8, train=0.8, ceiling=0.81, random=0.5, nd=0.85, duels=None, hybrid=None):
    out = {"test": (nd, test), "train": (nd, train), "ceiling": (nd, ceiling), "random": (0.45, random)}
    if duels is not None:
        out["duels"], out["hybrid"] = (nd, duels), (nd, hybrid)
    return out


def test_ceiling_verdict_quotes_the_numbers():
    e1 = {(0.0, 5): _cell(0.92), (0.0, 80): _cell(0.98), (0.7, 5): _cell(0.78), (0.7, 80): _cell(0.807, ceiling=0.809), (1.4, 5): _cell(0.66), (1.4, 80): _cell(0.697)}
    t = ev.verdict_ceiling(e1)[1]
    assert "zu **0,81** richtig geordnet (Zufall 0,50)" in t and "Obergrenze mit der wahren Funktion ist 0,81" in t and "(σ_u = 0): 0,98" in t and "(σ_u = 1,4): 0,70" in t
    assert "Mit 5 Gruppen sind es 0,78 statt 0,81" in t


def test_groups_verdict_quotes_the_numbers():
    e2 = {(8, 5): _cell(0.84, 0.86), (8, 20): _cell(0.85, 0.85), (8, 80): _cell(0.855, 0.85), (30, 5): _cell(0.767, 0.888), (30, 20): _cell(0.835, 0.859), (30, 80): _cell(0.846, 0.854)}
    t = ev.verdict_groups(e2)[1]
    assert "**0,89** auf den Trainingsgruppen, **0,77** auf neuen" in t and "Mit 80 Gruppen: 0,85 gegen 0,85" in t and "(0,86 gegen 0,84 bei 5 Gruppen, 0,85 gegen 0,85 bei 80)" in t


def test_losses_verdict_names_the_weakest_loss_and_the_spread():
    e3 = {(l, 5): _cell(0.78) for l in C.LOSSES}
    e3.update({(l, 80): _cell(0.806) for l in C.LOSSES})
    e3[(NDCG, 5)] = _cell(0.752)
    e3[(NDCG, 80)] = _cell(0.804)
    e3[(PAAR, 80)] = _cell(0.807)
    t = ev.verdict_losses(e3)[1]
    assert "innerhalb von **0,003**" in t and "(von 0,804 bis 0,807)" in t and "„NDCG-gewichtet“ am schwächsten (0,752 gegen 0,780 beim paarweisen Duellverlust)" in t
    e3[(STUFEN, 5)] = _cell(0.70)
    assert "„Paarweise“ am schwächsten (0,700" in ev.verdict_losses(e3)[1]


def test_warm_verdict_quotes_the_numbers():
    e4 = {(0.7, 0.5): _cell(0.80, duels=0.79, hybrid=0.83), (0.7, 1.0): _cell(0.80, duels=0.85, hybrid=0.86), (1.4, 1.0): _cell(0.69, duels=0.88, hybrid=0.87)}
    t = ev.verdict_warm(e4)[1]
    assert "Merkmale 0,80, Duelle allein 0,79, **Hybrid 0,83**" in t and "Mit allen Paaren: 0,80 / 0,85 / 0,86" in t and "(0,88 gegen Hybrid 0,87)" in t


def test_nonlinear_verdict_quotes_the_numbers():
    e5 = {(LIN, 20): _cell(0.74), (LIN, 80): _cell(0.75), (LIN, 200): _cell(0.75, ceiling=0.844), (NET, 20): _cell(0.74, train=0.86), (NET, 80): _cell(0.82), (NET, 200): _cell(0.84, ceiling=0.844)}
    t = ev.verdict_nonlinear(e5)[1]
    assert "bleibt das lineare Modell bei **0,75** (Obergrenze 0,84)" in t
    assert "Das Netz erreicht 0,84 mit 200 Gruppen und 0,82 mit 80, aber nur 0,74 mit 20 Gruppen (Training 0,86)" in t


def test_analysis_verdict_branches():
    from types import SimpleNamespace
    def make(q_test, train, test, hybrid=False):
        m = {"test": (0.8, test), "train": (0.8, train), "ceiling": (0.8, 0.81), "random": (0.45, 0.5)}
        if hybrid:
            m["duels"], m["hybrid"] = (0.8, 0.79), (0.8, 0.83)
        return SimpleNamespace(settings=Settings(q_test=q_test), metrics=m, explained=0.67)
    t = ev.verdict_analysis(make(0.0, 0.80, 0.80))[1]
    assert "ohne Partien" in t and "**0,800**" in t and "Obergrenze mit der wahren Funktion 0,810" in t and "Zufall 0,500" in t and "67 %" in t and "Überanpassung" not in t
    over = ev.verdict_analysis(make(0.0, 0.90, 0.80))[1]
    assert "Überanpassung" in over and "0,10 besser" in over
    assert "Überanpassung" not in ev.verdict_analysis(make(0.0, 0.829, 0.80))[1]                 # Lücke 0,029: unter der Schwelle von 0,03
    assert "Überanpassung" in ev.verdict_analysis(make(0.0, 0.84, 0.80))[1]
    warm = ev.verdict_analysis(make(0.5, 0.8, 0.8, hybrid=True))[1]
    assert "5,5 Spielen je Spieler" in warm and "Duelle allein 0,790, Hybrid (Merkmale als Prior) 0,830" in warm


def test_duels_and_hybrid_are_computed_for_every_test_group(monkeypatch):
    import ltr_hybrid as H
    calls = {"duels": 0, "hybrid": 0}
    real_d, real_h = H.duels_only, H.hybrid

    def d(W, lam=0.1):
        calls["duels"] += 1
        return real_d(W, lam)

    def h(W, prior, grid=C.LAM_GRID):
        calls["hybrid"] += 1
        return real_h(W, prior, grid)

    monkeypatch.setattr(H, "duels_only", d)
    monkeypatch.setattr(H, "hybrid", h)
    ev.analyse(Settings(q_groups=6, q_test=0.5, seed=8))
    assert calls == {"duels": C.N_TEST_GROUPS, "hybrid": C.N_TEST_GROUPS}
    calls.update(duels=0, hybrid=0)
    ev.run(8, C.TRUTHS[0], 0.7, 6, 1, PAAR, 0, 0.01, 8, q_test=0.5, n_test=7, warm=True)
    assert calls == {"duels": 7, "hybrid": 7}
    calls.update(duels=0, hybrid=0)
    ev._warm_run(0.7, 8)
    assert calls == {"duels": len(C.EXP_WARM_FRACTIONS) * C.N_WARM_GROUPS, "hybrid": len(C.EXP_WARM_FRACTIONS) * C.N_WARM_GROUPS}


def test_learning_curve_points_are_the_subsets_up_to_the_chosen_number():
    a = ev.analyse(Settings(q_groups=30, seed=2))
    assert [c[0] for c in a.curve] == [3, 6, 12, 25, 30]
    assert [c[0] for c in ev.analyse(Settings(q_groups=25, seed=2)).curve] == [3, 6, 12, 25]
