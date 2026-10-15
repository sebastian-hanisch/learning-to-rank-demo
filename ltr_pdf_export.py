"""PDF-Export der aktuellen Konfiguration und der Kennzahlen (fpdf2, Kernschrift: nur Zeichen aus Latin-1)."""

from fpdf import FPDF

import ltr_evaluation as ev


def _latin(text: str) -> str:
    for old, new in (("–", "-"), ("—", "-"), ("τ", "Tau"), ("σ", "sigma"), ("λ", "lambda"), ("µ", "mu"), ("θ", "theta"), ("≈", "ca."), ("≥", ">="), ("≤", "<="),
                     ("→", "->"), ("**", ""), ("€", "EUR"), ("−", "-"), ("„", '"'), ("“", '"'), ("f̂", "f"), ("̂", "")):
        text = text.replace(old, new)
    return text.encode("latin-1", "replace").decode("latin-1")


def build_pdf(a: ev.Analysis) -> bytes:
    s = a.settings
    pdf = FPDF()
    pdf.add_page()
    pdf.set_auto_page_break(auto=True, margin=15)
    pdf.set_font("Helvetica", "B", 15)
    pdf.multi_cell(0, 8, _latin("Learning to Rank: Rangfolge aus Merkmalen statt aus Duellen"), new_x="LMARGIN", new_y="NEXT")
    pdf.set_font("Helvetica", "", 10)
    pdf.ln(2)
    pdf.multi_cell(0, 5.5, _latin(f"Merkmale: {s.p}, Unerklärtes sigma_u: {ev.de(s.sigma_u, 1)}, Wahrheit: {s.truth}, Trainingsgruppen: {s.q_groups} (je {s.k_train} Duell je Paar), "
                                  f"Verlust: {s.loss}, Modell: {s.model}, Anteil gespielter Paare in den Testgruppen: {ev.de(s.q_test)}, Seed: {s.seed}."), new_x="LMARGIN", new_y="NEXT")
    pdf.ln(2)
    _level, text = ev.verdict_analysis(a)
    pdf.set_font("Helvetica", "B", 10)
    pdf.cell(0, 6, "Kernaussage", new_x="LMARGIN", new_y="NEXT")
    pdf.set_font("Helvetica", "", 10)
    pdf.multi_cell(0, 5.5, _latin(text), new_x="LMARGIN", new_y="NEXT")
    pdf.ln(3)
    headers = ("Bewertung", "NDCG@5", "Paartreffer")
    widths = (80, 30, 30)
    pdf.set_font("Helvetica", "B", 9)
    for h, w in zip(headers, widths):
        pdf.cell(w, 6, h, border=1)
    pdf.ln()
    pdf.set_font("Helvetica", "", 9)
    names = {"test": "Modell auf neuen Gruppen", "train": "Modell auf Trainingsgruppen", "ceiling": "Obergrenze (wahre Funktion)", "random": "Zufall", "duels": "Duelle allein", "hybrid": "Hybrid"}
    for key in ("test", "train", "ceiling", "random", "duels", "hybrid"):
        if key not in a.metrics:
            continue
        nd, pa = a.metrics[key]
        for value, w in zip((names[key], f"{nd:.3f}", f"{pa:.3f}"), widths):
            pdf.cell(w, 6, _latin(value), border=1)
        pdf.ln()
    pdf.ln(4)
    pdf.set_font("Helvetica", "I", 8)
    pdf.multi_cell(0, 4.5, _latin(
        "Alles ist simuliert (Merkmale, Stärken und Duelle aus dem Generator der Ranking-Linie). Die Obergrenze ist die Ordnung nach der wahren Funktion der Merkmale; was die Merkmale "
        "nicht erklären, holt kein Verfahren aus Merkmalen. Quelle: Demo von Sebastian Hanisch, sebastianhanisch.net."), new_x="LMARGIN", new_y="NEXT")
    return bytes(pdf.output())
