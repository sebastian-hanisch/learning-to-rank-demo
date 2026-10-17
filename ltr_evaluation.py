"""Analyse des gezeigten Laufs, fünf Experimente über die festen Läufe (SWEEP_SEEDS) und die Verdikte.

Gelernt wird eine Bewertungsfunktion f̂(x) aus Q Trainingsgruppen (je 12 Spieler, Duelle) und auf NEUEN Gruppen geprüft: NDCG@5 mit den Stufen der wahren Stärke und Paartreffer
(Anteil richtig geordneter Paare). Dazu immer die Obergrenze (Ordnung nach der wahren Funktion f(x), ohne das Unerklärte) und der Zufall."""

import functools
import math
from dataclasses import dataclass

import numpy as np

import ltr_constants as C
import ltr_hybrid as H
import ltr_models as M
import ltr_scenario as SC
import turnier_generator as T
from ltr_scenario import Settings

NAN = float("nan")
_SALT_INIT, _SALT_RANDOM = 0x1A17, 0x4A2D
_TEST_OFFSET = 1_000_000
LIN, NET = C.MODELS
PUNKT, PAAR, STUFEN, LIST, NDCG = C.LOSSES


def de(x: float, digits: int = 2) -> str:
    return f"{x:.{digits}f}".replace(".", ",")


def pct(x: float, digits: int = 0) -> str:
    return de(100 * x, digits)


def _mean(values) -> float:
    v = [x for x in values if not (isinstance(x, float) and math.isnan(x))]
    return float(sum(v) / len(v)) if v else NAN


def hidden_of(model: str) -> int:
    return C.HIDDEN if model == NET else 0


# --- Ein Lauf --------------------------------------------------------------------------------------------------------------------------------

@functools.lru_cache(maxsize=1024)
def _truth(p, truth, sigma_u, seed):
    return SC.make_truth(p, truth, sigma_u, seed)


@functools.lru_cache(maxsize=512)
def _train(p, truth, sigma_u, count, k, seed):
    return SC.make_groups(_truth(p, truth, sigma_u, seed), count, k, seed, 0)


@functools.lru_cache(maxsize=512)
def _test(p, truth, sigma_u, seed, q, count):
    return SC.make_groups(_truth(p, truth, sigma_u, seed), count, 1, seed, _TEST_OFFSET, q)


def _init(seed: int, p: int, hidden: int):
    if not hidden:
        return None
    return SC.normals(T.SplitMix64(int(seed) ^ _SALT_INIT), p * hidden + hidden)


def _random_scores(seed: int, shape) -> np.ndarray:
    return SC.normals(T.SplitMix64(int(seed) ^ _SALT_RANDOM), int(np.prod(shape))).reshape(shape)


def fit_for(p, truth, sigma_u, count, k, loss, hidden, l2, seed):
    tr = _train(p, truth, sigma_u, count, k, seed)
    return M.fit(tr.X, tr.W, loss, hidden, l2, _init(seed, p, hidden)), tr


def run(p, truth, sigma_u, count, k, loss, hidden, l2, seed, q_test=0.0, n_test=C.N_TEST_GROUPS, warm=False) -> dict:
    """Ein Lauf: anpassen, auf neuen Gruppen prüfen. Rückgabe: Kennzahlen (NDCG@5, Paartreffer) für Test, Training, Obergrenze und Zufall; bei `warm` auch Duelle allein und Hybrid."""
    net, tr = fit_for(p, truth, sigma_u, count, k, loss, hidden, l2, seed)
    te = _test(p, truth, sigma_u, seed, q_test, n_test)
    out = {"test": M.evaluate_scores(net.score(te.X), te.s), "train": M.evaluate_scores(net.score(tr.X[: min(count, 20)]), tr.s[: min(count, 20)]),
           "ceiling": M.evaluate_scores(te.f_true, te.s), "random": M.evaluate_scores(_random_scores(seed, te.s.shape), te.s)}
    if warm:
        f = net.score(te.X)
        duel = np.array([H.duels_only(te.W[g]) for g in range(n_test)])
        hyb = np.array([H.hybrid(te.W[g], f[g])[0] for g in range(n_test)])
        out["duels"] = M.evaluate_scores(duel, te.s)
        out["hybrid"] = M.evaluate_scores(hyb, te.s)
    return out


# --- Der gezeigte Lauf ------------------------------------------------------------------------------------------------------------------------

@dataclass(frozen=True)
class Analysis:
    settings: Settings
    test: SC.Groups
    scores: np.ndarray          # (n_test, m) Bewertung der Testgruppen
    metrics: dict               # test / train / ceiling / random (NDCG@5, Paartreffer), bei q_test > 0 auch duels / hybrid
    curve: list                 # (Q, Test-Paartreffer, Trainings-Paartreffer) für wachsende Gruppenzahlen
    explained: float            # Anteil der Streuung der Stärke, den f erklärt


def explained_share(sigma_u: float) -> float:
    return 1.0 / (1.0 + sigma_u ** 2)


def analyse(s: Settings) -> Analysis:
    hidden = hidden_of(s.model)
    net, tr = fit_for(s.p, s.truth, s.sigma_u, s.q_groups, s.k_train, s.loss, hidden, s.l2, s.seed)
    te = _test(s.p, s.truth, s.sigma_u, s.seed, s.q_test, C.N_TEST_GROUPS)
    f = net.score(te.X)
    metrics = {"test": M.evaluate_scores(f, te.s), "train": M.evaluate_scores(net.score(tr.X[: min(s.q_groups, 20)]), tr.s[: min(s.q_groups, 20)]),
               "ceiling": M.evaluate_scores(te.f_true, te.s), "random": M.evaluate_scores(_random_scores(s.seed, te.s.shape), te.s)}
    if s.q_test > 0:
        metrics["duels"] = M.evaluate_scores(np.array([H.duels_only(te.W[g]) for g in range(len(f))]), te.s)
        metrics["hybrid"] = M.evaluate_scores(np.array([H.hybrid(te.W[g], f[g])[0] for g in range(len(f))]), te.s)
    grid = sorted({q for q in (3, 6, 12, 25, 50, 100, 200, 300) if q < s.q_groups} | {s.q_groups})
    curve = []
    for q in grid:
        sub = _train(s.p, s.truth, s.sigma_u, q, s.k_train, s.seed)
        n2 = M.fit(sub.X, sub.W, s.loss, hidden, s.l2, _init(s.seed, s.p, hidden))
        curve.append((q, M.evaluate_scores(n2.score(te.X), te.s)[1], M.evaluate_scores(n2.score(sub.X[:20]), sub.s[:20])[1]))
    return Analysis(s, te, f, metrics, curve, explained_share(s.sigma_u))


def verdict_analysis(a: Analysis):
    s, m = a.settings, a.metrics
    text = (f"Neue Gruppen ({'ohne Partien' if s.q_test == 0 else f'mit {de(11 * s.q_test, 1)} Spielen je Spieler'}): Paartreffer **{de(m['test'][1], 3)}** (NDCG@5 {de(m['test'][0], 3)}); "
            f"Obergrenze mit der wahren Funktion {de(m['ceiling'][1], 3)}, Zufall {de(m['random'][1], 3)}. Die Merkmale erklären {pct(a.explained)} % der Streuung der Stärke. ")
    gap = m["train"][1] - m["test"][1]
    if gap > 0.03:
        text += f"Auf den Trainingsgruppen {de(m['train'][1], 3)}: das Modell ist {de(gap, 2)} besser als auf neuen Gruppen (Überanpassung). "
    if "hybrid" in m:
        text += f"Duelle allein {de(m['duels'][1], 3)}, Hybrid (Merkmale als Prior) {de(m['hybrid'][1], 3)}."
    return "info", text.strip()


# --- Feste Läufe der Experimente --------------------------------------------------------------------------------------------------------------

def _cell(p, truth, sigma_u, count, k, loss, hidden, l2=C.DEFAULT_L2, q_test=0.0, warm=False, n_test=C.N_TEST_GROUPS) -> dict:
    res = [run(p, truth, sigma_u, count, k, loss, hidden, l2, sd, q_test, n_test, warm) for sd in C.SWEEP_SEEDS]
    keys = res[0].keys()
    return {k: tuple(float(np.mean([r[k][i] for r in res])) for i in (0, 1)) for k in keys}


def sweep_ceiling() -> dict:
    """Experiment 1: Unerklärtes σ_u und Gruppenzahl Q (linear, paarweise, k = 1)."""
    return {(sg, q): _cell(C.DEFAULT_P, C.TRUTHS[0], sg, q, 1, PAAR, 0) for sg in C.EXP_SIGMAS for q in C.EXP_QS_SHORT}


def sweep_groups() -> dict:
    """Experiment 2: Gruppenzahl und viele Merkmale; Test gegen Training (σ_u = 0,5, k = 4)."""
    return {(p, q): _cell(p, C.TRUTHS[0], 0.5, q, 4, PAAR, 0) for p in (C.DEFAULT_P, C.EXP_P_MANY) for q in C.EXP_QS}


def sweep_losses() -> dict:
    """Experiment 3: die fünf Verluste (linear, σ_u = 0,7, k = 1)."""
    return {(loss, q): _cell(C.DEFAULT_P, C.TRUTHS[0], C.EXP_SIGMA, q, 1, loss, 0) for loss in C.LOSSES for q in C.EXP_QS_SHORT}


def _warm_run(sigma_u: float, seed: int) -> dict:
    """Ein Modell (Q = 40, paarweise, linear) je Lauf, geprüft auf Testgruppen mit wachsendem Anteil gespielter Paare."""
    net, tr = fit_for(C.DEFAULT_P, C.TRUTHS[0], sigma_u, C.EXP_WARM_Q, 1, PAAR, 0, C.DEFAULT_L2, seed)
    out = {}
    for q in C.EXP_WARM_FRACTIONS:
        te = _test(C.DEFAULT_P, C.TRUTHS[0], sigma_u, seed, q, C.N_WARM_GROUPS)
        f = net.score(te.X)
        out[q] = {"test": M.evaluate_scores(f, te.s), "ceiling": M.evaluate_scores(te.f_true, te.s), "random": M.evaluate_scores(_random_scores(seed, te.s.shape), te.s),
                  "duels": M.evaluate_scores(np.array([H.duels_only(te.W[g]) for g in range(len(f))]), te.s),
                  "hybrid": M.evaluate_scores(np.array([H.hybrid(te.W[g], f[g])[0] for g in range(len(f))]), te.s)}
    return out


def sweep_warm() -> dict:
    """Experiment 4: Merkmale, Duelle allein und Hybrid nach Anteil gespielter Paare der Testgruppen (Q = 40, paarweise, linear)."""
    out = {}
    for sg in C.EXP_SIGMAS:
        runs = [_warm_run(sg, sd) for sd in C.SWEEP_SEEDS]
        for q in C.EXP_WARM_FRACTIONS:
            out[(sg, q)] = {k: tuple(float(np.mean([r[q][k][i] for r in runs])) for i in (0, 1)) for k in runs[0][q]}
    return out


def sweep_nonlinear() -> dict:
    """Experiment 5: nichtlineare Wahrheit, linear gegen Netz (σ_u = 0,5, k = 4, paarweise)."""
    return {(model, q): _cell(C.DEFAULT_P, C.TRUTHS[1], 0.5, q, 4, PAAR, hidden_of(model)) for model in C.MODELS for q in C.EXP_NL_QS}


# --- Verdikte ---------------------------------------------------------------------------------------------------------------------------------

def verdict_ceiling(e1):
    low, mid, high = (e1[(sg, 80)]["test"][1] for sg in C.EXP_SIGMAS)
    ceil_mid = e1[(0.7, 80)]["ceiling"][1]
    return "info", (f"Neue Spieler ohne ein Duell werden bei 80 Trainingsgruppen zu **{de(mid, 2)}** richtig geordnet (Zufall {de(e1[(0.7, 80)]['random'][1])}); die Obergrenze mit der wahren "
                    f"Funktion ist {de(ceil_mid, 2)}. Bleibt nichts unerklärt (σ_u = 0): {de(low, 2)}; bei viel Unerklärtem (σ_u = 1,4): {de(high, 2)}. Mit 5 Gruppen sind es "
                    f"{de(e1[(0.7, 5)]['test'][1], 2)} statt {de(mid, 2)}: Mehr Gruppen holen die Obergrenze heran, übersteigen sie im Mittel nicht.")


def verdict_groups(e2):
    lo5, lo20, lo80 = (e2[(C.DEFAULT_P, q)] for q in C.EXP_QS)
    hi5 = e2[(C.EXP_P_MANY, 5)]
    return "warning", (f"Mit {C.EXP_P_MANY} Merkmalen und 5 Trainingsgruppen liegen Training und Test weit auseinander: **{de(hi5['train'][1], 2)}** auf den Trainingsgruppen, "
                       f"**{de(hi5['test'][1], 2)}** auf neuen. Mit 80 Gruppen: {de(e2[(C.EXP_P_MANY, 80)]['train'][1], 2)} gegen {de(e2[(C.EXP_P_MANY, 80)]['test'][1], 2)}. Bei "
                       f"{C.DEFAULT_P} Merkmalen ist die Lücke klein ({de(lo5['train'][1], 2)} gegen {de(lo5['test'][1], 2)} bei 5 Gruppen, {de(lo80['train'][1], 2)} gegen "
                       f"{de(lo80['test'][1], 2)} bei 80). Wer auf den Trainingsgruppen prüft, überschätzt das Modell.")


def verdict_losses(e3):
    q5 = {l: e3[(l, 5)]["test"][1] for l in C.LOSSES}
    q80 = {l: e3[(l, 80)]["test"][1] for l in C.LOSSES}
    spread80 = max(q80.values()) - min(q80.values())
    worst5 = min(q5, key=q5.get)
    return "info", (f"Bei 80 Trainingsgruppen liegen alle fünf Verluste innerhalb von **{de(spread80, 3)}** Paartreffer (von {de(min(q80.values()), 3)} bis {de(max(q80.values()), 3)}). "
                    f"Bei 5 Gruppen ist „{worst5.split(' (')[0]}“ am schwächsten ({de(q5[worst5], 3)} gegen {de(q5[PAAR], 3)} beim paarweisen Duellverlust): "
                    "Bei linearem Modell und linearer Wahrheit entscheidet der Verlust hier wenig, die Zahl der Gruppen mehr.")


def verdict_warm(e4):
    r = e4[(0.7, 0.5)]
    r1 = e4[(0.7, 1.0)]
    r14 = e4[(1.4, 1.0)]
    return "info", (f"Mit der Hälfte der Paare gespielt (rund 5,5 Spiele je Spieler, σ_u = 0,7): Merkmale {de(r['test'][1], 2)}, Duelle allein {de(r['duels'][1], 2)}, **Hybrid "
                    f"{de(r['hybrid'][1], 2)}**. Mit allen Paaren: {de(r1['test'][1], 2)} / {de(r1['duels'][1], 2)} / {de(r1['hybrid'][1], 2)}. Bei viel Unerklärtem (σ_u = 1,4) "
                    f"tragen Duelle allein fast alles ({de(r14['duels'][1], 2)} gegen Hybrid {de(r14['hybrid'][1], 2)}). Die Merkmale sind der Prior, die Duelle korrigieren ihn.")


def verdict_nonlinear(e5):
    big = C.EXP_NL_QS[-1]
    lin = {q: e5[(LIN, q)]["test"][1] for q in C.EXP_NL_QS}
    net = {q: e5[(NET, q)]["test"][1] for q in C.EXP_NL_QS}
    ceil = e5[(NET, big)]["ceiling"][1]
    return "warning", (f"Bei nichtlinearer Wahrheit bleibt das lineare Modell bei **{de(lin[big], 2)}** (Obergrenze {de(ceil, 2)}). Das Netz erreicht {de(net[big], 2)} mit {big} Gruppen und "
                       f"{de(net[80], 2)} mit 80, aber nur {de(net[20], 2)} mit 20 Gruppen (Training {de(e5[(NET, 20)]['train'][1], 2)}): Ohne genug Gruppen überanpasst das Netz.")
