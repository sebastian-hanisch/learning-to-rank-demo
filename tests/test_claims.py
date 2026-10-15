"""JEDE Zahl und Aussage aus App-Texten und README wird hier nachgerechnet (über die festen Instanzen SWEEP_SEEDS, 60 Läufe; nie ein einzelner Seed).
Der Generator hängt nur von Seed und Position ab (SplitMix64), die Zahlen sind plattformunabhängig bis auf Gleitkommarand; Schwellen daher mit Abstand zur Messung."""

import pytest

import ltr_constants as C
import ltr_evaluation as ev

PUNKT, PAAR, STUFEN, LIST, NDCG = C.LOSSES
LIN, NET = C.MODELS


@pytest.fixture(scope="module")
def e1():
    return ev.sweep_ceiling()


@pytest.fixture(scope="module")
def e2():
    return ev.sweep_groups()


@pytest.fixture(scope="module")
def e3():
    return ev.sweep_losses()


@pytest.fixture(scope="module")
def e4():
    return ev.sweep_warm()


@pytest.fixture(scope="module")
def e5():
    return ev.sweep_nonlinear()


def acc(cell, key="test"):
    return cell[key][1]


# --- Experiment 1: Obergrenze -------------------------------------------------------------------------------------------------------------------

def test_features_reach_the_ceiling_that_the_unexplained_part_sets(e1):
    assert [acc(e1[(0.0, q)]) for q in (5, 80)] == pytest.approx([0.924, 0.982], abs=0.02)
    assert [acc(e1[(0.7, q)]) for q in (5, 80)] == pytest.approx([0.778, 0.807], abs=0.02)
    assert [acc(e1[(1.4, q)]) for q in (5, 80)] == pytest.approx([0.664, 0.697], abs=0.02)
    assert acc(e1[(0.7, 80)], "ceiling") == pytest.approx(0.809, abs=0.015) and acc(e1[(1.4, 80)], "ceiling") == pytest.approx(0.699, abs=0.015)
    assert acc(e1[(0.0, 80)], "ceiling") == 1.0
    for sg in C.EXP_SIGMAS:
        assert acc(e1[(sg, 80)]) > acc(e1[(sg, 5)])                                 # mehr Gruppen helfen
        assert acc(e1[(sg, 80)]) <= acc(e1[(sg, 80)], "ceiling") + 0.01              # die Obergrenze wird nicht überschritten
        assert 0.47 < acc(e1[(sg, 80)], "random") < 0.53
    assert acc(e1[(0.7, 80)], "ceiling") - acc(e1[(0.7, 80)]) < 0.01                  # bei 80 Gruppen an der Obergrenze
    assert e1[(0.7, 80)]["test"][0] == pytest.approx(0.851, abs=0.02)                 # NDCG@5


# --- Experiment 2: Gruppenzahl und Überanpassung ----------------------------------------------------------------------------------------------

def test_many_features_and_few_groups_overfit(e2):
    assert acc(e2[(30, 5)]) == pytest.approx(0.767, abs=0.025) and acc(e2[(30, 5)], "train") == pytest.approx(0.888, abs=0.025)
    assert acc(e2[(30, 5)], "train") - acc(e2[(30, 5)]) > 0.08
    assert acc(e2[(8, 5)], "train") - acc(e2[(8, 5)]) < 0.04
    assert acc(e2[(30, 80)], "train") - acc(e2[(30, 80)]) < 0.02
    assert [acc(e2[(30, q)]) for q in C.EXP_QS] == pytest.approx([0.767, 0.835, 0.846], abs=0.025)
    assert [acc(e2[(8, q)]) for q in C.EXP_QS] == pytest.approx([0.841, 0.852, 0.855], abs=0.02)


# --- Experiment 3: Verluste ---------------------------------------------------------------------------------------------------------------------

def test_the_loss_matters_little(e3):
    q80 = {l: acc(e3[(l, 80)]) for l in C.LOSSES}
    assert max(q80.values()) - min(q80.values()) < 0.01
    assert all(v == pytest.approx(0.806, abs=0.015) for v in q80.values())
    q5 = {l: acc(e3[(l, 5)]) for l in C.LOSSES}
    assert q5[PAAR] == pytest.approx(0.778, abs=0.02) and q5[PUNKT] == pytest.approx(0.777, abs=0.02)
    assert q5[NDCG] == pytest.approx(0.752, abs=0.025) and min(q5, key=q5.get) == NDCG
    assert q5[PAAR] - q5[NDCG] > 0.01
    assert abs(q5[PAAR] - q5[PUNKT]) < 0.01


# --- Experiment 4: warm -------------------------------------------------------------------------------------------------------------------------

def test_features_duels_and_the_hybrid(e4):
    f = acc(e4[(0.7, 0.5)])
    assert f == pytest.approx(0.799, abs=0.025)                                      # Merkmale: von den Duellen der Testgruppe unabhängig
    assert [acc(e4[(0.7, q)], "duels") for q in C.EXP_WARM_FRACTIONS] == pytest.approx([0.658, 0.714, 0.787, 0.848], abs=0.03)
    assert [acc(e4[(0.7, q)], "hybrid") for q in C.EXP_WARM_FRACTIONS] == pytest.approx([0.798, 0.807, 0.829, 0.861], abs=0.03)
    for q in C.EXP_WARM_FRACTIONS[1:]:
        assert acc(e4[(0.7, q)], "hybrid") >= acc(e4[(0.7, q)], "duels") - 0.005
    assert acc(e4[(0.7, 0.1)], "duels") < acc(e4[(0.7, 0.1)])                        # wenige Spiele: Merkmale besser als Duelle
    assert acc(e4[(0.7, 1.0)], "duels") > acc(e4[(0.7, 1.0)])                        # viele Spiele: Duelle besser als Merkmale
    assert acc(e4[(0.7, 0.5)], "hybrid") > max(acc(e4[(0.7, 0.5)]), acc(e4[(0.7, 0.5)], "duels")) + 0.02
    assert acc(e4[(1.4, 1.0)], "duels") == pytest.approx(0.876, abs=0.03) and acc(e4[(1.4, 1.0)], "hybrid") == pytest.approx(0.877, abs=0.03)
    assert acc(e4[(1.4, 1.0)], "duels") > acc(e4[(1.4, 1.0)]) + 0.1
    assert acc(e4[(0.0, 0.1)]) == pytest.approx(0.975, abs=0.02) and acc(e4[(0.0, 0.1)], "hybrid") < acc(e4[(0.0, 0.1)])      # nichts unerklärt: Duelle stören nur


# --- Experiment 5: nichtlinear --------------------------------------------------------------------------------------------------------------------

def test_the_net_needs_enough_groups(e5):
    assert [acc(e5[(LIN, q)]) for q in C.EXP_NL_QS] == pytest.approx([0.740, 0.748, 0.750], abs=0.02)
    assert [acc(e5[(NET, q)]) for q in C.EXP_NL_QS] == pytest.approx([0.740, 0.825, 0.835], abs=0.025)
    assert acc(e5[(NET, 20)], "train") == pytest.approx(0.860, abs=0.03)
    assert acc(e5[(NET, 20)], "train") - acc(e5[(NET, 20)]) > 0.08
    assert acc(e5[(NET, 80)]) > acc(e5[(LIN, 80)]) + 0.05 and acc(e5[(NET, 20)]) <= acc(e5[(LIN, 20)]) + 0.02
    assert acc(e5[(NET, 200)], "ceiling") == pytest.approx(0.844, abs=0.015) and acc(e5[(NET, 200)]) > acc(e5[(NET, 200)], "ceiling") - 0.025


# --- Verdikte nennen die Zahlen aus den Daten ---------------------------------------------------------------------------------------------------

def test_verdicts_quote_the_measured_numbers(e1, e2, e3, e4, e5):
    t = ev.verdict_ceiling(e1)[1]
    assert "zu **0,81** richtig geordnet" in t and "(σ_u = 0): 0,98" in t and "(σ_u = 1,4): 0,70" in t and "Mit 5 Gruppen sind es 0,78 statt 0,81" in t
    g = ev.verdict_groups(e2)[1]
    assert "Mit 30 Merkmalen und 5 Trainingsgruppen" in g and "**0,89** auf den Trainingsgruppen" in g and "**0,77** auf neuen" in g
    lo = ev.verdict_losses(e3)[1]
    assert "alle fünf Verluste innerhalb von **0,003**" in lo or "alle fünf Verluste innerhalb von **0,00" in lo
    assert "„NDCG-gewichtet“ am schwächsten" in lo
    w = ev.verdict_warm(e4)[1]
    assert "Merkmale 0,80, Duelle allein 0,79, **Hybrid 0,83**" in w
    n = ev.verdict_nonlinear(e5)[1]
    assert "bleibt das lineare Modell bei **0,75**" in n and "(Obergrenze 0,84)" in n and "mit 200 Gruppen" in n
