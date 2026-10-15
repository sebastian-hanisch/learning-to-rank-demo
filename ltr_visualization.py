"""Plotly-Abbildungen. Alle Achsen sind gesperrt (fixedrange), damit Touch-Geräte beim Scrollen nicht am Diagramm hängen bleiben."""

import numpy as np
import plotly.graph_objects as go

import ltr_constants as C
import ltr_evaluation as ev

SHORT_LOSS = {l: l.split(" (")[0] for l in C.LOSSES}


def lock_axes(fig):
    fig.update_xaxes(fixedrange=True)
    fig.update_yaxes(fixedrange=True)
    return fig


def _base(fig, height=380):
    fig.update_layout(height=height, margin=dict(l=10, r=10, t=10, b=10), legend=dict(orientation="h", y=-0.3), plot_bgcolor="rgba(0,0,0,0)")
    return lock_axes(fig)


def build_scatter(a: ev.Analysis):
    """Geschätzte gegen wahre Stärke aller neuen Spieler der ersten Testgruppen; die Diagonale wäre die perfekte Ordnung."""
    te = a.test
    groups = min(len(a.scores), 6)
    x = te.s[:groups].ravel()
    y = a.scores[:groups].ravel()
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=x, y=y, mode="markers", name="neue Spieler", marker=dict(color=C.COLORS[a.settings.model], size=7, opacity=0.75),
                             hovertemplate="wahre Stärke %{x:.2f}, Bewertung %{y:.2f}<extra></extra>"))
    lo, hi = float(x.min()), float(x.max())
    fig.add_trace(go.Scatter(x=[lo, hi], y=[lo * float(np.std(y) / np.std(x)), hi * float(np.std(y) / np.std(x))], mode="lines", name="gleiche Ordnung",
                             line=dict(color="rgba(150,150,150,0.8)", dash="dash")))
    fig.update_xaxes(title_text="wahre Stärke (Elo-Punkte-Maß θ)")
    fig.update_yaxes(title_text="Bewertung f̂(x)")
    return _base(fig, height=380)


def build_group_ranks(a: ev.Analysis):
    """Eine Testgruppe: wahrer Rang gegen den Rang nach der Bewertung (ein Punkt je Spieler)."""
    s, f = a.test.s[0], a.scores[0]
    true_rank = np.argsort(np.argsort(-s)) + 1
    est_rank = np.argsort(np.argsort(-f, kind="stable")) + 1
    order = np.argsort(true_rank)
    fig = go.Figure()
    fig.add_trace(go.Bar(x=[str(r) for r in true_rank[order]], y=est_rank[order], marker_color=C.COLORS[a.settings.model],
                         hovertemplate="wahrer Rang %{x}: geordnet auf Platz %{y}<extra></extra>"))
    fig.add_trace(go.Scatter(x=[str(r) for r in true_rank[order]], y=true_rank[order], mode="markers", name="wahrer Rang", marker=dict(color=C.COLOR_TRUTH, size=9, symbol="diamond")))
    fig.update_xaxes(title_text="Spieler nach wahrem Rang (1 = Bester)", type="category")
    fig.update_yaxes(title_text="Platz nach der Bewertung", autorange="reversed")
    fig.update_layout(showlegend=True)
    return _base(fig, height=340)


def build_learning_curve(a: ev.Analysis):
    qs = [c[0] for c in a.curve]
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=qs, y=[c[1] for c in a.curve], mode="lines+markers", name="neue Gruppen (Test)", line=dict(color="#2F6B65", width=3)))
    fig.add_trace(go.Scatter(x=qs, y=[c[2] for c in a.curve], mode="lines+markers", name="Trainingsgruppen", line=dict(color="#c0392b", width=2, dash="dot")))
    fig.add_hline(y=a.metrics["ceiling"][1], line_dash="dash", line_color=C.COLOR_TRUTH, annotation_text="Obergrenze", annotation_position="bottom right")
    fig.add_hline(y=0.5, line_dash="dot", line_color="rgba(150,150,150,0.8)", annotation_text="Zufall", annotation_position="top right")
    fig.update_xaxes(title_text="Trainingsgruppen", type="category")
    fig.update_yaxes(title_text="Paartreffer", range=[0.4, 1.0])
    return _base(fig, height=340)


def build_ceiling(e1):
    fig = go.Figure()
    sig = list(C.EXP_SIGMAS)
    for q, color in zip(C.EXP_QS_SHORT, ("#9cc5bf", "#2F6B65")):
        fig.add_trace(go.Bar(x=[f"σ_u = {s}".replace(".", ",") for s in sig], y=[e1[(s, q)]["test"][1] for s in sig], name=f"{q} Trainingsgruppen", marker_color=color,
                             hovertemplate="%{x}, " + str(q) + " Gruppen: %{y:.3f}<extra></extra>"))
    fig.add_trace(go.Scatter(x=[f"σ_u = {s}".replace(".", ",") for s in sig], y=[e1[(s, C.EXP_QS_SHORT[-1])]["ceiling"][1] for s in sig], mode="markers", name="Obergrenze (wahre Funktion)",
                             marker=dict(color=C.COLOR_TRUTH, size=12, symbol="diamond")))
    fig.add_hline(y=0.5, line_dash="dot", line_color="rgba(150,150,150,0.8)", annotation_text="Zufall", annotation_position="bottom right")
    fig.update_layout(barmode="group")
    fig.update_yaxes(title_text="Paartreffer auf neuen Gruppen", range=[0.4, 1.0])
    return _base(fig, height=360)


def build_groups(e2):
    fig = go.Figure()
    for p, color in ((C.DEFAULT_P, "#2F6B65"), (C.EXP_P_MANY, "#c0392b")):
        fig.add_trace(go.Scatter(x=[str(q) for q in C.EXP_QS], y=[e2[(p, q)]["test"][1] for q in C.EXP_QS], mode="lines+markers", name=f"{p} Merkmale, neue Gruppen",
                                 line=dict(color=color, width=3)))
        fig.add_trace(go.Scatter(x=[str(q) for q in C.EXP_QS], y=[e2[(p, q)]["train"][1] for q in C.EXP_QS], mode="lines+markers", name=f"{p} Merkmale, Trainingsgruppen",
                                 line=dict(color=color, width=2, dash="dot")))
    fig.update_xaxes(title_text="Trainingsgruppen", type="category")
    fig.update_yaxes(title_text="Paartreffer", range=[0.7, 0.95])
    return _base(fig, height=360)


def build_losses(e3):
    fig = go.Figure()
    for loss in C.LOSSES:
        fig.add_trace(go.Bar(x=[f"{q} Gruppen" for q in C.EXP_QS_SHORT], y=[e3[(loss, q)]["test"][1] for q in C.EXP_QS_SHORT], name=SHORT_LOSS[loss], marker_color=C.LOSS_COLORS[loss],
                             hovertemplate=SHORT_LOSS[loss] + ", %{x}: %{y:.3f}<extra></extra>"))
    fig.update_layout(barmode="group")
    fig.update_yaxes(title_text="Paartreffer auf neuen Gruppen", range=[0.7, 0.85])
    return _base(fig, height=360)


def build_warm(e4, sigma: float = 0.7):
    xs = [f"{11 * q:.1f}".replace(".", ",") for q in C.EXP_WARM_FRACTIONS]
    fig = go.Figure()
    for key, name, color in (("test", "Merkmale", C.COLORS["Merkmale"]), ("duels", "Duelle allein", C.COLORS["Duelle allein"]), ("hybrid", "Hybrid", C.COLORS["Hybrid"])):
        fig.add_trace(go.Scatter(x=xs, y=[e4[(sigma, q)][key][1] for q in C.EXP_WARM_FRACTIONS], mode="lines+markers", name=name, line=dict(color=color, width=3)))
    fig.add_hline(y=e4[(sigma, C.EXP_WARM_FRACTIONS[0])]["ceiling"][1], line_dash="dash", line_color=C.COLOR_TRUTH, annotation_text="Obergrenze der Merkmale", annotation_position="bottom right")
    fig.update_xaxes(title_text="Spiele je Spieler in den neuen Gruppen", type="category")
    fig.update_yaxes(title_text="Paartreffer", range=[0.5, 1.0])
    return _base(fig, height=360)


def build_nonlinear(e5):
    fig = go.Figure()
    for model in C.MODELS:
        fig.add_trace(go.Scatter(x=[str(q) for q in C.EXP_NL_QS], y=[e5[(model, q)]["test"][1] for q in C.EXP_NL_QS], mode="lines+markers", name=f"{model}, neue Gruppen",
                                 line=dict(color=C.COLORS[model], width=3)))
        fig.add_trace(go.Scatter(x=[str(q) for q in C.EXP_NL_QS], y=[e5[(model, q)]["train"][1] for q in C.EXP_NL_QS], mode="lines+markers", name=f"{model}, Trainingsgruppen",
                                 line=dict(color=C.COLORS[model], width=2, dash="dot")))
    fig.add_hline(y=e5[(C.MODELS[0], C.EXP_NL_QS[0])]["ceiling"][1], line_dash="dash", line_color=C.COLOR_TRUTH, annotation_text="Obergrenze", annotation_position="top right")
    fig.update_xaxes(title_text="Trainingsgruppen", type="category")
    fig.update_yaxes(title_text="Paartreffer", range=[0.65, 0.9])
    return _base(fig, height=360)
