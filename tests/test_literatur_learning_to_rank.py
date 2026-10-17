"""Literatur-Referenzwerte zu Learning to Rank (ORACLE-PLAYBOOK §5.1). Nur Werte und Formeln, die in der genannten Quelle eingesehen wurden; keiner stammt aus dem Code der Demo.

Wert 1 (NDCG-Beispiel): sechs Dokumente mit den Relevanzen 3, 2, 3, 0, 1, 2 in der Reihenfolge der Suchmaschine; DCG_6 = 6.861; ideale Ordnung über alle bekannten Urteile
  (zwei weitere Dokumente mit 3 und 2) 3, 3, 3, 2, 2, 2 mit IDCG_6 = 8.740; nDCG_6 = 6.861 / 8.740 = 0.785. Gewinn = rel (linear), Abschlag 1 / log2(i + 1), wie in `ndcg_at`.
  Quelle: Wikipedia, „Discounted cumulative gain“, Abschnitt Beispiel, https://en.wikipedia.org/wiki/Discounted_cumulative_gain, eingesehen am 2026-10-17 (zwei Abrufe,
  gleiche Zahlen). Die Aufgabe gilt nur für die Konvention log2(i + 1); die Demo folgt Burges (2010), nicht Järvelin/Kekäläinen (2002), siehe unten.

Wert 2 (RankNet-Kosten): C = log(1 + e^{−(s_i − s_j)}) für ein Paar, bei dem i höher zu ordnen ist (S_ij = 1, σ = 1); „Note that when s_i = s_j, the cost is log 2“.
  Quelle: Burges, C. J. C. (2010): From RankNet to LambdaRank to LambdaMART: An Overview. Microsoft Research Technical Report MSR-TR-2010-82, Abschnitt 2 (RankNet), Seite 3,
  der Satz nach der Kostenformel C = ½(1 − S_ij)(s_i − s_j) + log(1 + e^{−(s_i − s_j)}); Text aus der PDF-Fassung extrahiert und gelesen, eingesehen am 2026-10-17. Die Formel
  für andere Abstände ist aus derselben Gleichung von Hand ausgewertet (kein eigener Literaturwert).

Wert 3 (ListNet, Formel der Quelle, von Hand ausgewertet): Top-eins-Wahrscheinlichkeit P_s(j) = exp(s_j) / Σ_k exp(s_k) und Kreuzentropie L(y, z) = − Σ_j P_y(j) log P_z(j).
  Quelle: Cao, Z., Qin, T., Liu, T.-Y., Tsai, M.-F., Li, H. (2007): Learning to rank: from pairwise approach to listwise approach. ICML 2007 (MSR-TR-2007-40), Abschnitt 4.2
  (Definition 5, Theorem 6) und Abschnitt 4.2 „listwise loss function“ mit Kreuzentropie, https://www.microsoft.com/en-us/research/wp-content/uploads/2016/02/tr-2007-40.pdf,
  PDF gelesen am 2026-10-17. Das Beispiel der Rechnung ist selbst gewählt; es zählt als von Hand gerechnet, nicht als Literaturzahl.

Nicht übereinstimmbar (kein Test): Järvelin, K., Kekäläinen, J. (2002), ACM TOIS 20(4), 422–446, Abschnitt 2.2, Gleichung (2): DCG[i] = DCG[i − 1] + G[i] / log_b i erst ab Rang b,
  Beispiel G = 3, 2, 3, 0, 0, 1, 2, 2, 3, 0 mit b = 2 liefert DCG = 3, 5, 6.89, 6.89, 6.89, 7.28, 7.99, 8.66, 9.61, 9.61 (eingesehen am 2026-10-17). Die Demo verwendet den Abschlag
  1 / log2(i + 1) ab Rang 1 und Gewinne 2^Stufe − 1 (Burges 2010, Gleichung (5)); die Zahlen der Quelle lassen sich mit der Demo nicht nachrechnen."""

import math

import numpy as np
import pytest

import ltr_constants as C
import ltr_models as M


def test_ndcg_matches_the_worked_example_of_the_dcg_reference():
    rel = np.array([3.0, 2.0, 3.0, 0.0, 1.0, 2.0, 3.0, 2.0])                   # D1 bis D6 in der Reihenfolge der Suchmaschine, dann die zwei nicht gefundenen Urteile 3 und 2
    scores = np.array([8.0, 7.0, 6.0, 5.0, 4.0, 3.0, 2.0, 1.0])                # Ordnung der Suchmaschine; die zwei weiteren Dokumente stehen dahinter
    assert M.ndcg_at(scores, rel, 6) == pytest.approx(0.785, abs=5e-4)          # 6.861 / 8.740


def test_pairwise_cost_is_log_two_for_equal_scores_and_log_one_plus_exp_minus_difference_otherwise():
    for gap in (0.0, 2.0, -1.5):                                              # s_i − s_j; i hat gewonnen
        f = np.array([[gap, 0.0]])
        W = np.array([[[0.0, 1.0], [0.0, 0.0]]])
        pts = W.sum(axis=2)
        expected = math.log(1.0 + math.exp(-gap))
        n = 2                                                                  # der Verlust ist durch die Zahl der Spieler geteilt
        assert n * M.total_loss(C.LOSSES[1], f, W, pts, M.grades_from(pts), 1.0) == pytest.approx(expected)
        assert n * M.total_loss(C.LOSSES[2], f, W, pts, M.grades_from(pts), 1.0) == pytest.approx(expected)       # RankNet auf Stufen: derselbe Paarverlust


def test_listwise_loss_is_the_cross_entropy_of_the_top_one_probabilities():
    W = np.array([[[0.0, 1.0, 1.0], [0.0, 0.0, 1.0], [0.0, 0.0, 0.0]]])        # Siege 2, 1, 0 -> zentriert (1, 0, −1) bei Streuung 1
    pts = W.sum(axis=2)
    f = np.array([[0.5, 0.0, -0.25]])
    target = np.exp([1.0, 0.0, -1.0])
    target = target / target.sum()                                             # P_y(j) = exp(y_j) / Σ exp(y_k)
    z = np.exp(f[0])
    z = z / z.sum()                                                            # P_z(j)
    expected = -float(np.sum(target * np.log(z)))
    assert 3 * M.total_loss(C.LOSSES[3], f, W, pts, M.grades_from(pts), 1.0) == pytest.approx(expected)
