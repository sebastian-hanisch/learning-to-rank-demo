"""Learning to Rank: Rangfolge aus Merkmalen statt aus Duellen - interaktive Konzept-Demo
Sebastian Hanisch - Operations Research und Machine Learning

Vierzehntes Stück der Ranking-Linie: Spieler haben Merkmale; gelernt wird eine Funktion, die neue Spieler ordnet, auch ohne ein einziges Duell. Fünf Experimente messen, wie weit
Merkmale tragen (Obergrenze durch das Unerklärte), wann man überanpasst, ob der Verlust zählt (punktweise, paarweise, listenweise, NDCG-gewichtet), was Duelle gegen Merkmale bringen
(und der Hybrid) und wann ein Netz nötig ist. Aufbau und Regeln: DEMO-PLAYBOOK.md (§2).

Lauffähig mit: streamlit run app.py
"""

import pandas as pd
import streamlit as st

import ltr_constants as C
import ltr_evaluation as ev
from ltr_pdf_export import build_pdf
from ltr_presets import (
    apply_preset,
    bounds,
    init_session_state_defaults,
    load_permalink_settings,
    randomize_seed,
    sync_query_params,
)
from ltr_scenario import Settings
from ltr_visualization import (
    SHORT_LOSS,
    build_ceiling,
    build_group_ranks,
    build_groups,
    build_learning_curve,
    build_losses,
    build_nonlinear,
    build_scatter,
    build_warm,
)

st.set_page_config(page_title="Learning to Rank – Sebastian Hanisch", layout="wide")

SHOW = {"success": st.success, "info": st.info, "warning": st.warning}


def verdict(level, text):
    SHOW[level](text)


@st.cache_data(show_spinner=False)
def _analysis(settings):
    return ev.analyse(settings)


@st.cache_data(show_spinner="Experiment rechnet über 60 feste Läufe …")
def _ceiling():
    return ev.sweep_ceiling()


@st.cache_data(show_spinner="Experiment rechnet über 60 feste Läufe …")
def _groups():
    return ev.sweep_groups()


@st.cache_data(show_spinner="Experiment rechnet über 60 feste Läufe …")
def _losses():
    return ev.sweep_losses()


@st.cache_data(show_spinner="Experiment rechnet über 60 feste Läufe …")
def _warm():
    return ev.sweep_warm()


@st.cache_data(show_spinner="Experiment rechnet über 60 feste Läufe …")
def _nonlinear():
    return ev.sweep_nonlinear()


st.title("🧮 Learning to Rank: Rangfolge aus Merkmalen statt aus Duellen")

load_permalink_settings()
init_session_state_defaults()

with st.sidebar:
    st.header("⚙️ Einstellungen")
    p = st.slider("Merkmale je Spieler", *bounds("p_slider"), key="p_slider", help="Wie viele beobachtbare Merkmale (Größe, Form, Vorjahr …) jeder Spieler hat.")
    sigma = st.slider("Unerklärtes σ_u", *bounds("sigma_slider"), step=C.SIGMA_STEP, key="sigma_slider", format="%.1f",
                      help="Streuung der Stärke, die die Merkmale nicht erklären (Funktion f(x) hat die Streuung 1). 0 = die Merkmale erklären alles.")
    truth = st.selectbox("Wahrheit", C.TRUTHS, key="truth_select", help="Wie die Stärke von den Merkmalen abhängt: linear oder nichtlinear (tanh, sin, Wechselwirkung).")
    q_groups = st.slider("Trainingsgruppen", *bounds("q_slider"), key="q_slider", help="Wie viele Saisons (je 12 Spieler mit Duellen) zum Lernen vorliegen.")
    k_train = st.slider("Duelle je Paar im Training", *bounds("k_slider"), key="k_slider", help="Wie oft jedes Paar einer Trainingsgruppe gegeneinander gespielt hat.")
    loss = st.selectbox("Verlust", C.LOSSES, key="loss_select", help="Womit das Modell trainiert wird: Regression auf die Siege, Duell-Logistik, RankNet auf Stufen, ListNet, NDCG-gewichtet.")
    model = st.selectbox("Modell", C.MODELS, key="model_select", help="Linear oder ein kleines Netz mit einer verdeckten Schicht.")
    q_test = st.select_slider("Gespielte Paare in neuen Gruppen", C.Q_TEST_CHOICES, key="qtest_select",
                              help="Anteil der Paare, die in den neuen Gruppen schon gespielt wurden. 0 = Kaltstart (keine Partie); mehr zeigt Duelle allein und den Hybrid.")
    l2 = st.slider("Ridge (Gewichtsstrafe)", *bounds("l2_slider"), step=C.L2_STEP, key="l2_slider", format="%.4f", help="Strafe auf die Parameter des Modells.")
    seed = st.number_input("Zufalls-Seed", *bounds("seed_input"), key="seed_input", step=1)
    st.button("🎲 Neuen Lauf würfeln", width="stretch", on_click=randomize_seed)

sync_query_params({"p_slider": int(p), "sigma_slider": round(float(sigma), 1), "truth_select": truth, "q_slider": int(q_groups), "k_slider": int(k_train), "loss_select": loss,
                   "model_select": model, "qtest_select": float(q_test), "l2_slider": round(float(l2), 4), "seed_input": int(seed)})

st.markdown(
    """
Wer zum ersten Mal spielt, hat keine Duelle, aber Merkmale. **Learning to Rank** lernt aus früheren Gruppen („Saisons“) eine Funktion, die Spieler allein nach ihren Merkmalen ordnet. Diese
Demo ist das **vierzehnte Stück der Ranking-Linie** und misst, **wie weit Merkmale tragen**, **was der Verlust ausmacht**, **wann Duelle besser sind** und **wann man ein Netz braucht**.
Immer mit der **Obergrenze** daneben: Was die Merkmale nicht erklären, holt kein Verfahren aus Merkmalen.

Weiter unten: fünf **🔬 Experimente** mit Verdikt, **🚧 Wo die Annahmen enden** und der Expander **📐 Mathematische Formulierung**.
"""
)

with st.expander("Was heißt Rangfolge aus Merkmalen?", expanded=True):
    st.markdown(
        """
1. **Gruppen:** Eine Saison sind 12 Spieler, die gegeneinander spielen. Beobachtet werden ihre Merkmale und wie oft jeder gewonnen hat; die wahre Stärke kennt nur die Simulation.
2. **Bewertungsfunktion:** Gelernt wird f̂(x), eine Zahl je Spieler aus den Merkmalen. Wer die größte Zahl hat, steht oben.
3. **Verluste:** *Punktweise* regrediert die Siege auf die Merkmale; *paarweise* lernt aus jedem Duell („wer gewinnt, hat die größere Zahl“); *listenweise* vergleicht die ganze Gruppe;
   *NDCG-gewichtet* gewichtet Fehler an der Spitze stärker.
4. **Prüfung:** auf **neuen** Gruppen ohne Partien; **NDCG@5** (stehen die Besten oben?) und **Paartreffer** (Anteil richtig geordneter Paare).
5. **Hybrid:** Sind in den neuen Gruppen schon Duelle gespielt, nimmt Bradley-Terry die Merkmalsbewertung als Prior und lässt die Duelle korrigieren.
        """
    )

st.caption("🎯 Schnellstart – ein Beispielszenario laden:")
cols = st.columns(len(C.PRESETS))
for col, name in zip(cols, C.PRESETS):
    with col:
        st.button(name, width="stretch", on_click=apply_preset, args=(name,), help=C.PRESET_HELP.get(name, ""), key=f"preset_{name}")

st.caption("🔗 Die Adresszeile oben spiegelt Ihre aktuelle Konfiguration wider – einfach kopieren, um ein Szenario zu teilen.")

settings = Settings(int(p), round(float(sigma), 1), truth, int(q_groups), int(k_train), loss, model, float(q_test), round(float(l2), 4), int(seed))
a = _analysis(settings)

st.markdown("---")

# --- Ergebnis ------------------------------------------------------------------------------------------------------------------------------

st.markdown("## 🧮 Die Rangfolge neuer Spieler")
m = a.metrics
show = [("Modell auf neuen Gruppen", "test"), ("Obergrenze", "ceiling"), ("Zufall", "random"), ("Modell auf Trainingsgruppen", "train")]
if "duels" in m:
    show += [("Duelle allein", "duels"), ("Hybrid", "hybrid")]
cols = st.columns(len(show))
for col, (label, key) in zip(cols, show):
    nd, pa = m[key]
    col.metric(label, ev.de(pa, 3), delta=f"NDCG@5 {ev.de(nd, 3)}", delta_color="off", help="Anteil richtig geordneter Spielerpaare (0,5 = Zufall); darunter NDCG@5.")
verdict(*ev.verdict_analysis(a))

cl, cr = st.columns(2)
with cl:
    st.plotly_chart(build_scatter(a), width="stretch", key="scatter_chart")
    st.caption("Bewertung gegen wahre Stärke der Spieler aus sechs neuen Gruppen (ein Punkt je Spieler); je enger an der Linie, desto besser die Ordnung.")
with cr:
    st.plotly_chart(build_group_ranks(a), width="stretch", key="rank_chart")
    st.caption("Eine neue Gruppe: Platz nach der Bewertung (Balken) gegen den wahren Rang (Raute).")
st.plotly_chart(build_learning_curve(a), width="stretch", key="curve_chart")
st.caption("Paartreffer über die Zahl der Trainingsgruppen (dieselben Gruppen als Teilmengen): auf neuen Gruppen und auf den Trainingsgruppen selbst.")
st.download_button("📄 PDF des aktuellen Laufs", data=build_pdf(a), file_name="learning-to-rank.pdf", mime="application/pdf", key="pdf_download")

st.markdown("---")

# --- Experimente ---------------------------------------------------------------------------------------------------------------------------

st.subheader("🔬 Experiment 1: Wie weit tragen Merkmale?")
st.caption("8 Merkmale, lineares Modell mit paarweisem Duellverlust, 60 feste Läufe, 40 neue Gruppen je Lauf; σ_u ist das Unerklärte. Die Raute ist die Ordnung nach der wahren Funktion.")
e1 = _ceiling()
st.plotly_chart(build_ceiling(e1), width="stretch", key="exp1_chart")
verdict(*ev.verdict_ceiling(e1))

st.subheader("🔬 Experiment 2: Wie viele Gruppen, und wann überanpasst man?")
st.caption(f"σ_u = 0,5, 4 Duelle je Paar; {C.DEFAULT_P} gegen {C.EXP_P_MANY} Merkmale. Gestrichelt: Paartreffer auf den Trainingsgruppen.")
e2 = _groups()
st.plotly_chart(build_groups(e2), width="stretch", key="exp2_chart")
verdict(*ev.verdict_groups(e2))

st.subheader("🔬 Experiment 3: Zählt der Verlust?")
st.caption("Lineares Modell, σ_u = 0,7, ein Duell je Paar im Training; die fünf Verluste auf denselben Gruppen.")
e3 = _losses()
st.plotly_chart(build_losses(e3), width="stretch", key="exp3_chart")
verdict(*ev.verdict_losses(e3))
st.dataframe(pd.DataFrame([{"Verlust": SHORT_LOSS[l], **{f"{q} Gruppen: Paartreffer": round(e3[(l, q)]["test"][1], 3) for q in C.EXP_QS_SHORT},
                            **{f"{q} Gruppen: NDCG@5": round(e3[(l, q)]["test"][0], 3) for q in C.EXP_QS_SHORT}} for l in C.LOSSES]), hide_index=True, width="stretch")

st.subheader("🔬 Experiment 4: Merkmale, Duelle allein oder Hybrid?")
st.caption("40 Trainingsgruppen; in den neuen Gruppen ist ein Anteil der Paare einmal gespielt (11 mal Anteil = Spiele je Spieler). Hybrid: Bradley-Terry mit der Merkmalsbewertung als Prior.")
e4 = _warm()
c1, c2 = st.columns(2)
with c1:
    st.markdown("**σ_u = 0,7**")
    st.plotly_chart(build_warm(e4, 0.7), width="stretch", key="exp4a_chart")
with c2:
    st.markdown("**σ_u = 1,4**")
    st.plotly_chart(build_warm(e4, 1.4), width="stretch", key="exp4b_chart")
verdict(*ev.verdict_warm(e4))

st.subheader("🔬 Experiment 5: Wann braucht man ein Netz?")
st.caption("Nichtlineare Wahrheit (tanh, sin, Wechselwirkung), σ_u = 0,5, 4 Duelle je Paar, paarweiser Verlust.")
e5 = _nonlinear()
st.plotly_chart(build_nonlinear(e5), width="stretch", key="exp5_chart")
verdict(*ev.verdict_nonlinear(e5))

st.markdown("---")

# --- Grenzen -------------------------------------------------------------------------------------------------------------------------------

st.subheader("🚧 Wo die Annahmen enden")
st.markdown(
    """
| Annahme | Was passiert, wenn sie verletzt ist | Wer setzt an |
|---|---|---|
| **Die Merkmale sind relevant und vollständig genug** | Was sie nicht erklären (σ_u), bleibt: die Obergrenze liegt bei 0,81 für σ_u = 0,7 (Experiment 1). Duelle sind dann der einzige Weg. | Bradley-Terry |
| **Eine Funktion für alle Gruppen** | Gruppeneffekte (Liga, Saison, Platz) sind nicht modelliert; das Modell würde sie in die Merkmale pressen. | – |
| **Stufen kommen aus den Siegen** | Echte Relevanzstufen kommen von Bewertern, die selbst unsicher sind. | Die Bewerter sind unsicher |
| **Merkmale unabhängig, keine fehlenden Werte** | Korrelierte, fehlende oder irrelevante Merkmale verschlechtern die Schätzung; nicht gemessen. | – |
| **Eigene Umsetzung der Verluste** | Die Ranking-Ziele der Boosting-Bibliotheken (LightGBM, XGBoost) sind nicht gemessen und nicht gegen deren Dokumentation geprüft. | – |
| **12 Spieler je Gruppe, 8 Merkmale** | Andere Größen nur über die Regler, nicht in den festen Läufen. | – |
| **Alles ist simuliert** | Die Zahlen gelten für diese Wahrheit und dieses Rauschen. | – |
"""
)

st.markdown("---")

with st.expander("📐 Mathematische Formulierung"):
    st.markdown(
        r"""
**Wahrheit.** Spieler mit Merkmalen $x \in \mathbb{R}^p$, $x \sim N(0, I)$; Stärke $s = f(x) + \sigma_u u$, $u \sim N(0,1)$; $f(x) = w^\top x$ ($\lVert w\rVert = 1$) oder
$f(x) = \bigl[0{,}8\tanh(2 w^\top x) + 0{,}6\sin(1{,}5\, v^\top x) + 0{,}4\, x_1 x_2\bigr]/0{,}9$. Duell: $P(i \text{ schlägt } j) = \sigma(s_i - s_j)$. Eine Gruppe: $m = 12$ neue Spieler.

**Modell.** $\hat f(x) = \beta^\top x$ oder $w_2^\top \tanh(W_1 x + b_1)$ mit 8 verdeckten Einheiten.

**Verluste** (je Gruppe, $N$ = Zahl aller Spieler):
- *punktweise:* $\tfrac1N\sum_i (\hat f_i - \tilde y_i)^2$ mit den zentrierten, skalierten Siegen $\tilde y_i$;
- *paarweise (Duelle):* $-\tfrac1N\sum_{i \ne j} W_{ij}\ln\sigma(\hat f_i - \hat f_j)$ (Bradley-Terry mit Merkmalen);
- *paarweise (Stufen, RankNet):* $-\tfrac1N\sum_{g_i > g_j}\ln\sigma(\hat f_i - \hat f_j)$ mit den Stufen $g \in \{0,\dots,4\}$ aus den Siegen;
- *listenweise (ListNet):* $-\tfrac1N\sum_i t_i \ln \operatorname{softmax}(\hat f)_i$ mit $t = \operatorname{softmax}(\tilde y)$;
- *NDCG-gewichtet:* wie RankNet, jedes Paar mit $|G_i - G_j|\,\bigl|1/\log_2(r_i + 1) - 1/\log_2(r_j + 1)\bigr|$ ($G = 2^g - 1$, $r$ der aktuelle Rang), Faktor 3.

Training: Adam, 150 Schritte, Schrittweite 0,1, plus $\ell_2$ · Parameter; keine Konvergenzprüfung.

**Maße.** $\mathrm{NDCG@5} = \mathrm{DCG@5}/\mathrm{IDCG@5}$ mit Gewinnen $2^{g} - 1$ aus den Stufen der wahren Stärke und $\mathrm{DCG} = \sum_k G_{(k)}/\log_2(k+1)$. *Paartreffer:* Anteil der Paare,
die $\hat f$ wie $s$ ordnet (Gleichstand ½). *Obergrenze:* dieselben Maße für die Ordnung nach $f(x)$.

**Hybrid.** Bradley-Terry je Gruppe mit Prior $s_i \sim N(\hat f(x_i), 1/\lambda_u)$; $\lambda_u$ aus $\{0{,}3, \dots, 100\}$ nach der Laplace-Evidenz
$\ln p(D \mid \lambda) \approx \ln \mathrm{post}(\hat s) + \tfrac{n}{2}\ln\lambda - \tfrac12\ln\det H$.

**Zufall.** SplitMix64-Strom des Generators (Box-Muller für die Normalwerte); je Gruppe ein eigener Strom aus (Seed, Nummer), Gruppe $g$ hängt nur davon ab.

**Literatur.**
- Burges, C. et al. (2005): Learning to rank using gradient descent. *ICML 2005* (RankNet; nicht eingesehen).
- Cao, Z., Qin, T., Liu, T.-Y., Tsai, M.-F., Li, H. (2007): Learning to rank: from pairwise approach to listwise approach. *ICML 2007* (ListNet; nicht eingesehen).
- Burges, C. J. C. (2010): From RankNet to LambdaRank to LambdaMART: an overview. *Microsoft Research Technical Report MSR-TR-2010-82* (nicht eingesehen).
- Järvelin, K., Kekäläinen, J. (2002): Cumulated gain-based evaluation of IR techniques. *ACM Transactions on Information Systems* 20, 422–446 (NDCG; nicht eingesehen).
- Bradley, R. A., Terry, M. E. (1952): Rank analysis of incomplete block designs. *Biometrika* 39, 324–345 (nicht eingesehen).
- Steele, G. L., Lea, D., Flood, C. H. (2014): Fast splittable pseudorandom number generators. *OOPSLA 2014*, 453–472 (Zufallsstrom des Generators).
        """
    )

st.markdown("---")
st.caption(
    "Diese Demo ist Teil des Portfolios von [Sebastian Hanisch](https://sebastianhanisch.net) – "
    "Operations Research und Machine Learning ([Über mich](https://sebastianhanisch.net/ueber-mich.html)). "
    "Mehr zur Reihe: [Ranking: von der Siegtabelle bis Learning to Rank](https://sebastianhanisch.net/konzepte-ranking.html)."
)
