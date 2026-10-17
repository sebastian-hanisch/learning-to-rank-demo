"""Konstanten: Regler-Grenzen, feste Annahmen, Farben und Presets."""

M_GROUP = 12                                  # Spieler je Gruppe (Saison)
P_MIN, P_MAX, DEFAULT_P = 2, 20, 8
SIGMA_MIN, SIGMA_MAX, SIGMA_STEP, DEFAULT_SIGMA = 0.0, 2.0, 0.1, 0.7
TRUTHS = ("Linear", "Nichtlinear")
DEFAULT_TRUTH = TRUTHS[0]
Q_MIN, Q_MAX, DEFAULT_Q = 3, 300, 40
K_MIN, K_MAX, DEFAULT_K = 1, 8, 1
LOSSES = ("Punktweise (Regression)", "Paarweise (Duelle)", "Paarweise (Stufen, RankNet)", "Listenweise (ListNet)", "NDCG-gewichtet (LambdaRank-artig)")
DEFAULT_LOSS = LOSSES[1]
MODELS = ("Linear", "Netz (8 verdeckte Einheiten)")
DEFAULT_MODEL = MODELS[0]
HIDDEN = 8
Q_TEST_CHOICES = (0.0, 0.1, 0.25, 0.5, 1.0)   # Anteil der Paare, die in den Testgruppen schon gespielt wurden (0 = Kaltstart)
DEFAULT_Q_TEST = Q_TEST_CHOICES[0]
L2_MIN, L2_MAX, L2_STEP, DEFAULT_L2 = 0.0001, 1.0, 0.0001, 0.01
SEED_MAX, DEFAULT_SEED = 999999, 3

ADAM_ITERS, ADAM_LR = 150, 0.1
NDCG_K = 5
N_TEST_GROUPS = 40                            # neue Gruppen je Lauf
N_WARM_GROUPS = 8                             # neue Gruppen je Lauf im Warmstart-Experiment (Hybrid rechnet je Gruppe)
LAM_GRID = (0.3, 1.0, 3.0, 10.0, 30.0, 100.0)    # Prior-Präzision des Hybrids

# Feste Instanzen für alle Experimente: nie ein einzelner Seed als Befund.
SWEEP_SEEDS = tuple(range(100000, 100060))
EXP_SIGMA = 0.7
EXP_QS = (5, 20, 80)
EXP_QS_SHORT = (5, 80)
EXP_SIGMAS = (0.0, 0.7, 1.4)
EXP_NL_QS = (20, 80, 200)
EXP_P_MANY = 30
EXP_WARM_Q = 40
EXP_WARM_FRACTIONS = (0.1, 0.25, 0.5, 1.0)

COLORS = {"Linear": "#2F6B65", "Netz (8 verdeckte Einheiten)": "#c0392b", "Merkmale": "#2F6B65", "Duelle allein": "#9aa0a6", "Hybrid": "#8e6bb0"}
COLOR_TRUTH = "#e67e22"
LOSS_COLORS = {LOSSES[0]: "#9aa0a6", LOSSES[1]: "#2F6B65", LOSSES[2]: "#4c78a8", LOSSES[3]: "#8e6bb0", LOSSES[4]: "#c0392b"}

_BASE = {"p": DEFAULT_P, "sigma_u": DEFAULT_SIGMA, "truth": DEFAULT_TRUTH, "q_groups": DEFAULT_Q, "k_train": DEFAULT_K, "loss": DEFAULT_LOSS, "model": DEFAULT_MODEL,
         "q_test": DEFAULT_Q_TEST, "l2": DEFAULT_L2, "seed": DEFAULT_SEED}
PRESETS = {
    "Neue Spieler ohne Duelle": dict(_BASE),
    "Wenn nichts unerklärt bleibt": {**_BASE, "sigma_u": 0.0},
    "Merkmale gegen Duelle": {**_BASE, "q_test": 0.5},
    "Das Netz lernt Nichtlineares": {**_BASE, "truth": TRUTHS[1], "sigma_u": 0.5, "q_groups": 80, "k_train": 4, "model": MODELS[1]},
}
PRESET_HELP = {
    "Neue Spieler ohne Duelle": "Acht Merkmale, 40 Trainingsgruppen: Neue Spieler ohne ein einziges Duell werden zu rund 0,8 richtig geordnet; das ist im Erwartungswert die Obergrenze dieser Merkmale.",
    "Wenn nichts unerklärt bleibt": "Die Merkmale erklären die Stärke vollständig: Neue Spieler werden fast fehlerfrei geordnet.",
    "Merkmale gegen Duelle": "Die Testgruppen haben die Hälfte der Paare gespielt (rund 5,5 Spiele je Spieler): Duelle allein, Merkmale allein und der Hybrid im Vergleich.",
    "Das Netz lernt Nichtlineares": "Nichtlineare Wahrheit, 80 Trainingsgruppen, mehrere Duelle je Paar: Das Netz holt, was das lineare Modell nicht sieht.",
}
