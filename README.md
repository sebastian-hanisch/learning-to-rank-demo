# Learning to Rank: Rangfolge aus Merkmalen statt aus Duellen – Streamlit-Demo

**[→ Demo live ausprobieren](https://sebastianhanisch-learning-to-rank-demo.streamlit.app/)**

Vierzehntes Stück der **Ranking-Linie** (Konzepte-Demo), Nachfolger von `rangintervalle-demo` und `bradley-terry-demo`: Wer zum ersten Mal spielt, hat keine Duelle, aber Merkmale.
**Learning to Rank** lernt aus früheren Gruppen („Saisons“) eine Funktion, die Spieler nach ihren Merkmalen ordnet. Gemessen wird, **wie weit Merkmale tragen** (Obergrenze durch das
Unerklärte), **wann man überanpasst**, **ob der Verlust zählt** (punktweise, paarweise, listenweise, NDCG-gewichtet), **was Duelle gegen Merkmale bringen** (und ein Hybrid) und
**wann ein Netz nötig ist**. Das Vehikel ist das neutrale, simulierte Turnier der Wurzel (Zufallsstrom aus dem Generator Version 1, wortgleich); alles ist simuliert.

## Kernfrage

Wie weit kommt man mit Merkmalen allein? **Acht Merkmale, die zwei Drittel der Streuung der Stärke erklären (Unerklärtes σ_u = 0,7):** neue Spieler ohne ein einziges Duell werden zu
**0,81** richtig geordnet (Zufall 0,50, NDCG@5 0,85), und das ist die **Obergrenze** (die Ordnung nach der wahren Funktion ohne Rauschen: 0,81). Mit 5 Trainingsgruppen sind es 0,78, mit 80
0,81. Bleibt nichts unerklärt: 0,98; bei σ_u = 1,4: 0,70. **Der Verlust zählt wenig:** bei 80 Gruppen liegen alle fünf innerhalb von 0,003, bei 5 Gruppen ist der NDCG-gewichtete am
schwächsten (0,75 gegen 0,78). **Duelle schlagen Merkmale erst mit genug Spielen:** bei 5,5 Spielen je Spieler Merkmale 0,80, Duelle allein 0,79, **Hybrid** (Bradley-Terry mit den Merkmalen
als Prior) **0,83**; bei 11 Spielen 0,80 / 0,85 / 0,86. **Ein Netz lohnt nur bei nichtlinearer Wahrheit und genug Gruppen:** linear 0,75, Netz 0,825 bei 80 und 0,835 bei 200 Gruppen (Obergrenze 0,844), aber
0,740 bei 20 (Training 0,860). Wer auf den Trainingsgruppen prüft, sieht bei 30 Merkmalen und 5 Gruppen 0,89 statt 0,77.

## Modell

- **Wahrheit:** Spieler mit p Merkmalen x ~ N(0, I); Stärke s = f(x) + σ_u · u, u ~ N(0, 1); f linear (|w| = 1, Streuung 1) oder nichtlinear (tanh, sin, Wechselwirkung). σ_u ist, was die
  Merkmale nicht erklären. Duell: P(i schlägt j) = σ(s_i − s_j).
- **Gruppen:** 12 neue Spieler je Saison; im Training je Paar k Partien, in den neuen Gruppen mit der Wahrscheinlichkeit q je Paar eine Partie (q = 0: Kaltstart).
- **Bewertungsfunktion:** linear oder ein Netz mit einer verdeckten Schicht (8 tanh-Einheiten); Adam, 150 Schritte, Schrittweite 0,1.
- **Verluste:** *punktweise* (Regression der zentrierten Siege), *paarweise* auf den Duellen (Bradley-Terry mit Merkmalen), *paarweise auf Stufen* (RankNet), *listenweise* (ListNet),
  *NDCG-gewichtet* (LambdaRank-artig, Gewicht |ΔGewinn| · |ΔAbschlag| des aktuellen Rangs).
- **Maße:** NDCG@5 mit Gewinnen 2^Stufe − 1 (Stufen 0 bis 4 aus der wahren Stärke), Paartreffer (Anteil richtig geordneter Paare); dazu immer die **Obergrenze** (Ordnung nach f(x)) und der Zufall.
- **Hybrid:** Bradley-Terry je Gruppe mit Prior N(f̂(x), 1/λ_u); λ_u aus {0,3 … 100} nach der Laplace-Evidenz (wie der Empirical Bayes in `rangintervalle-demo`).
- Alle Formeln stehen im 📐-Expander der App.

## Methodik

- **Experimente** über 60 feste Läufe (`SWEEP_SEEDS`, nie ein einzelner Seed), je mit einem live berechneten Verdikt; 12 Spieler je Gruppe, 8 Merkmale, 40 neue Gruppen je Lauf, unabhängig vom Regler.
- **Neue Gruppen:** Die Prüfung läuft auf Gruppen, die nie im Training waren; der Test auf den Trainingsgruppen wird nur als Lücke gezeigt.
- **Zufall:** `turnier_generator.py` mit SplitMix64-Strom (wortgleich mit den Repos der Linie); Normalwerte per Box-Muller; je Gruppe ein eigener Strom aus (Seed, Nummer), die ersten Q Gruppen
  bleiben bei größerem Q dieselben.
- **Rechenzeit:** Die Experimente rechnen beim ersten Aufruf rund anderthalb Minuten und werden danach zwischengespeichert.

## Befunde (gemessen, keine Behauptungen)

Alle Zahlen: 60 feste Läufe, 8 Merkmale, 12 Spieler je Gruppe, lineares Modell mit paarweisem Duellverlust, wo nichts anderes steht; Paartreffer auf neuen Gruppen (NDCG@5 in Klammern).

| Frage | Befund | Test |
|---|---|---|
| Obergrenze durch das Unerklärte (5 / 80 Trainingsgruppen) | σ_u = 0: **0,92 / 0,98** (0,96 / 0,99); σ_u = 0,7: 0,78 / **0,81** (0,82 / 0,85), Obergrenze 0,81; σ_u = 1,4: 0,66 / **0,70** (0,68 / 0,72), Obergrenze 0,70. Zufall 0,50. | `test_features_reach_the_ceiling_that_the_unexplained_part_sets` |
| Überanpassung bei 5 / 20 / 80 Gruppen, σ_u = 0,5, 4 Duelle je Paar | 8 Merkmale: Test 0,841 / 0,852 / 0,855, Training 0,858 / 0,851 / 0,850; **30 Merkmale: Test 0,767 / 0,835 / 0,846, Training 0,888 / 0,859 / 0,854**. | `test_many_features_and_few_groups_overfit` |
| Verluste (punktweise / Duelle / Stufen / ListNet / NDCG-gewichtet), 5 und 80 Gruppen | 5 Gruppen: 0,777 / 0,778 / 0,770 / 0,770 / **0,752**; 80 Gruppen: 0,807 / 0,807 / 0,807 / 0,807 / 0,804 (alle innerhalb 0,003). | `test_the_loss_matters_little` |
| Warm, σ_u = 0,7 (Merkmale / Duelle allein / Hybrid) bei 1,1 / 2,8 / 5,5 / 11 Spielen je Spieler | Merkmale 0,80 (unabhängig von den Spielen); Duelle 0,66 / 0,71 / 0,79 / **0,85**; Hybrid 0,80 / 0,81 / **0,83** / **0,86**. σ_u = 1,4, 11 Spiele: 0,69 / **0,88** / 0,88. σ_u = 0: Merkmale 0,975, Hybrid 0,94 bis 0,96. | `test_features_duels_and_the_hybrid` |
| Nichtlineare Wahrheit, σ_u = 0,5, 20 / 80 / 200 Gruppen (linear / Netz) | linear 0,740 / 0,748 / 0,750; Netz **0,740 / 0,825 / 0,835**, Training des Netzes bei 20 Gruppen 0,860; Obergrenze 0,844. | `test_the_net_needs_enough_groups` |
| Verdikte nennen dieselben Zahlen | Die live berechneten Texte der App geben die obigen Werte wieder. | `test_verdicts_quote_the_measured_numbers` |

Die Verluste unterscheiden sich bei 5 Gruppen um bis zu 0,026, bei 80 um 0,003; Unterschiede dieser Größe sind **kleiner als die Streuung zwischen Instanzen** und nicht als Rangfolge der
Verfahren zu lesen.

## Befunde und Korrekturen gegenüber dem Plan

- **Korrektur 2026-10-17 (Quellenangabe, Eigenschaftstest):** Die README nannte Järvelin/Kekäläinen (2002) für den NDCG, die Demo rechnet aber in der Schreibweise nach Burges 2010 (Gewinn 2^Stufe − 1, Abschlag 1/log₂(i+1)); die Quellenzeile sagt das jetzt. Die Rechnung selbst ist durch den Test gegen ein veröffentlichtes DCG-Beispiel und gegen Aufzählen mit `math.log2` belegt. Zahlen unverändert.

- **„Der Hybrid liegt fast immer oben“ stimmt nur beinahe:** Bei viel Unerklärtem (σ_u = 1,4) und 5,5 Spielen liegen Duelle allein (0,823) knapp vor dem Hybrid (0,817), und bei
  σ_u = 0 stören die Duelle die Merkmale (Hybrid 0,935 gegen Merkmale 0,975 bei 1,1 Spielen). Der Hybrid gewinnt dort, wo Merkmale und Duelle beide etwas wissen.
- **Gitter des Hybrids bis 100 statt 10:** In der Vorab-Messung verlor der Hybrid bei σ_u = 0,3 und 1,1 Spielen gegen die Merkmale, weil das Gitter nur bis 10 reichte; mit dem Gitter bis 100
  tritt das nur noch bei σ_u = 0 auf.
- **Netz bis 200 statt 300 Gruppen:** zur Rechenzeit; die Aussage bleibt (0,835 gegen 0,844 der Obergrenze).
- Die Zahlen verschieben sich um wenige Hundertstel gegen die Vorab-Messung (60 statt 30 Läufe, SplitMix64-Strom, 150 statt 300 Adam-Schritte bei gleicher Genauigkeit).

## Abgrenzung

- **Bradley-Terry** (Stück 5): Stärke je Spieler aus Duellen, keine Merkmale; hier eine Funktion der Merkmale, die auch neue Spieler ordnet.
- **Rangintervalle und Shrinkage** (Stück 13): Shrinkage zur Mitte (Prior-Mittel 0); hier das Prior-Mittel f̂(x) (Hybrid) mit derselben Evidenz.
- **Ordinale Regression** (`ordinale-regression-demo`, Linie Mehrkriterien): lernt per LP eine additive Wertfunktion aus wenigen exakt geordneten Referenzen (Zulässigkeit, deterministischer
  Entscheider); hier ein statistisches Lernproblem mit verrauschten Duellen, Verallgemeinerung auf neue Spieler und Ranking-Verlusten.
- **Von der Mannschaft zum Einzelnen** (Stück 10): Zerlegung von Teamergebnissen in Einzelstärken (Merkmal = Zugehörigkeit); hier freie Merkmale.
- **Nachfolger (nur beim Namen):** Die Bewerter sind unsicher; Welches Verfahren, gemessen woran?

## Ehrliche Grenzen

- **Die Merkmale sind gegeben, relevant und unabhängig (N(0, 1)):** Echte Merkmale sind korreliert, teils irrelevant, teils fehlend; nicht gemessen.
- **Eine Funktion für alle Gruppen:** keine Gruppeneffekte (Saison, Liga).
- **Stufen aus den Siegen:** In der Praxis kommen Stufen von Bewertern, die selbst unsicher sind.
- **Eigene Umsetzung der Verluste:** Die Ranking-Ziele von Boosting-Bibliotheken (LightGBM, XGBoost) sind nicht gemessen und nicht gegen deren Dokumentation geprüft.
- **12 Spieler je Gruppe, 8 Merkmale (30 im Überanpassungs-Experiment):** Andere Größen nur über die Regler.
- **Adam in fester Schrittzahl:** keine Konvergenzprüfung; das Netz ist klein (8 Einheiten) und mit einer Initialisierung je Lauf trainiert.
- **Formparameter der Sigmoidfunktion:** In der Duellwahrscheinlichkeit und im paarweisen Verlust ist der Formparameter der Sigmoidfunktion 1; Burges 2010 führt ihn als σ ein und lässt ihn frei.
- **Zielverteilung von ListNet:** Die Zielverteilung (Softmax der zentrierten Siege) ist eine Wahl der Demo; Cao et al. definieren sie über Zielwerte y, hier stehen die Siege an der Stelle von y.
- **Alles simuliert:** keine Aussage über einen Sport oder ein Suchsystem.

## Quellen

- Burges, C. et al. (2005): Learning to rank using gradient descent. *ICML 2005* (RankNet; eingesehen 2026-10-17, ICML-Fassung, Abschnitt 3, Gl. (1)–(3): Kreuzentropie auf Paaren mit logistischer Wahrscheinlichkeit, C_ij = −P̄_ij o_ij + log(1 + e^{o_ij}); Gl. (16): NDCG mit Gewinn 2^r − 1 und Abschlag 1/log(1 + j)).
- Cao, Z., Qin, T., Liu, T.-Y., Tsai, M.-F., Li, H. (2007): Learning to rank: from pairwise approach to listwise approach. *ICML 2007* (ListNet; eingesehen 2026-10-17, Fassung MSR-TR-2007-40, Abschnitt 4.2, Definition 5 und Theorem 6, und Abschnitt 5: Top-eins-Wahrscheinlichkeit P_s(j) = exp(s_j) / Σ_k exp(s_k), Kreuzentropie L(y, z) = −Σ_j P_y(j) log P_z(j)).
- Burges, C. J. C. (2010): From RankNet to LambdaRank to LambdaMART: an overview. *Microsoft Research Technical Report MSR-TR-2010-82* (eingesehen 2026-10-17, Abschnitt 2, S. 3: Kosten C = log(1 + e^{−(s_i − s_j)}) für ein Paar mit S_ij = 1, bei s_i = s_j gleich log 2; Abschnitt 3, Gl. (5): DCG@T = Σ (2^{l_i} − 1) / log(1 + i), die Basis des Logarithmus ist dort nicht angegeben und für den NDCG als Quotient ohne Wirkung; Gl. (6): LambdaRank gewichtet den RankNet-Gradienten mit |ΔNDCG|).
- Järvelin, K., Kekäläinen, J. (2002): Cumulated gain-based evaluation of IR techniques. *ACM Transactions on Information Systems* 20, 422–446 (Ursprung des NDCG; eingesehen 2026-10-17, Abschnitt 2.2, Gl. (2), und Abschnitt 2.3, Gl. (5): Gewinn = Relevanzstufe selbst, Abschlag 1/log_b i erst ab Rang b, Normierung durch das ideale DCG; die Demo rechnet NDCG mit Gewinn 2^Stufe − 1 und Abschlag 1/log₂(i+1), der Schreibweise nach Burges 2010 (Basis des Logarithmus dort offen, hier 2), nicht nach der Originaldefinition).
- Bradley, R. A., Terry, M. E. (1952): Rank analysis of incomplete block designs. *Biometrika* 39, 324–345 (nicht eingesehen).
- Steele, G. L., Lea, D., Flood, C. H. (2014): Fast splittable pseudorandom number generators. *OOPSLA 2014*, 453–472 (Zufallsstrom des Generators).
- **Nicht herangezogen:** die Dokumentation der Ranking-Ziele von LightGBM und XGBoost; die Zahlen dieser Demo sind Simulation.

## Tests

100 Tests, einige Minuten (die Experimente rechnen über 60 feste Läufe): Einheiten gegen Handwerte (`test_models.py`: Stufen, NDCG, Paartreffer, Bewertungsfunktion mit Rückwärtsrechnung gegen finite
Differenzen, alle fünf Verlustgradienten gegen finite Differenzen, Training; `test_scenario.py`: Wahrheit, Box-Muller, Gruppen, Duelle, Hybrid), Generator (`test_generator.py`, wortgleich),
**Orakel** (`test_oracle_learning_to_rank.py`: paarweise Anpassung gegen den BFGS von scipy, NDCG und Paartreffer gegen Aufzählen mit `math.log2`, Eigenschaften über feste Läufe: fast fehlerfrei
ohne Unerklärtes, nahe Zufall bei viel Unerklärtem, Lücke zwischen Training und Test, Hybrid gegen Duelle und gegen Merkmale), Maße und Verdikte (`test_evaluation.py`), alle Zahlen dieses README
(`test_claims.py`), Presets, Szenario und PDF (`test_presets.py`), Laufzeit-Importe und AppTest-Rauchtests.

**Mutationstest** (`tools/oracle_mutation.py` der Website, eingeschränkt auf die vier `ltr_`-Module; schnelle Testdateien zuerst, Überlebende gegen die Claims-Tests): **185 Stellen, 169 getötet, 16 äquivalent.** Im ersten, schnellen Lauf überlebten 48 Stellen (137 getötet); 32 davon sind gegen die Claims-Tests oder mit neuen Einheitentests getötet. **Echte Lücken** waren: die Gewinne 2^Stufe − 1 der NDCG-Auswertung, der Fall eines kleinen Gesamtgewinns (IDCG ≤ 1), die Adam-Rekursion (Schrittzahl, Bias-Korrektur, Skalierung der Zielwerte, jeweils gegen eine unabhängige Rekursion an fünf Verlusten und beiden Modellen), die Initialisierung des Netzes ohne Startwerte, die Zuordnung der Gruppennummern zu Strömen, die Gewinnhäufigkeit der Duelle, die Evidenz des Hybrids (gegen numerische Integration bei zwei und drei Spielern, das Vorzeichen der Hesse-Matrix zeigt sich erst bei drei) und die Zahl der Gruppen, für die Duelle und Hybrid gerechnet werden; sie sind mit Tests geschlossen und einzeln getötet. **16 sind äquivalent:** die Konstante in den Gewinnen des NDCG-Gewichts (nur Differenzen gehen ein), die Verschiebung um das Maximum in Softmax (mathematisch ohne Wirkung), die zehnte Newton-Iteration des Hybrids (schon konvergiert), `>` gegen `>=` bei Gleitkommagleichstand der Evidenz und am Rand 0,03 des Hinweises, `<` gegen `<=` bei stetigen Würfen, die zusätzliche Maske `& (2⁶⁴ − 1)` der Strom-Kennung (der Generator maskiert selbst) und der Fall k = 0 der Duelle.

## Dateistruktur

| Datei | Inhalt |
|---|---|
| `app.py` | Streamlit-Oberfläche |
| `turnier_generator.py` | gemeinsamer Generator (wortgleich in allen Repos der Linie) |
| `ltr_models.py` | Bewertungsfunktion (linear, Netz), fünf Verluste, Adam, NDCG, Paartreffer |
| `ltr_hybrid.py` | Bradley-Terry mit Prior (Duelle allein, Hybrid, Evidenz) |
| `ltr_scenario.py` | Reglerwerte (`Settings`), Wahrheit, Gruppen |
| `ltr_evaluation.py` | Analyse, fünf Experimente über die festen Läufe, Verdikte |
| `ltr_presets.py`, `ltr_constants.py` | Permalink, Presets, Grenzen |
| `ltr_visualization.py`, `ltr_pdf_export.py` | Plotly-Abbildungen, PDF-Export |
| `tests/` | Einheiten-, Orakel-, Claims- und App-Tests |

## Bewusst nicht umgesetzt

- **Boosting-Bibliotheken** (LightGBM, XGBoost) und deren Ranking-Ziele.
- **Gruppeneffekte, fehlende Werte, korrelierte Merkmale** und Merkmale aus Rohdaten.
- **Expertenstufen mit eigener Unsicherheit** (das Stück „Die Bewerter sind unsicher“ setzt dort an).

## Lokal ausführen

```bash
pip install -r requirements.txt
streamlit run app.py
```

Tests: `pip install -r requirements-dev.txt` (enthält scipy für das Orakel) und `python -m pytest tests -q`.

Gebaut mit Streamlit, Plotly, numpy, pandas und fpdf2.

---

Diese Demo ist Teil des Portfolios von [Sebastian Hanisch](https://sebastianhanisch.net) – Operations Research und Machine Learning ([Über mich](https://sebastianhanisch.net/ueber-mich.html)). Mehr zur Reihe: [Ranking: von der Siegtabelle bis Learning to Rank](https://sebastianhanisch.net/konzepte-ranking.html).
