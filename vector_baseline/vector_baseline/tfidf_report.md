# Report (baseline_tdidf, sanity_check_ranking, sanity_check_ranking_with_post)
## 1. Avg τ = 0.387 (Kendall’s Tau)

Tau misst, wie gut die relative Reihenfolge deiner vorhergesagten Ränge (rpred) mit der wahren Rangordnung (true_rank) übereinstimmt.

- Wertebereich: −1 → +1
- +1 = perfekte Ordnung
- 0 = Zufall
- −1 = komplett verkehrt herum

Dein Wert 0.387 → das Modell ordnet deutlich besser als Zufall, aber noch nicht perfekt.

**Es erkennt grob, welche Antworten besser sind, aber verwechselt gelegentlich Nachbarn oder Mittelplätze.**

## 2. Avg ρ = 0.430 (Spearman’s Rho)

Auch eine Rangkorrelation, aber weniger empfindlich gegenüber einzelnen Fehlpaaren. Typisch ist, dass ρ > τ, weil Spearman glatter ist — das ist bei dir der Fall. 0.43 bestätigt: 

**Ein klar positiver Zusammenhang, also das Ranking folgt der wahren Ordnung im Mittel recht konsistent, aber nicht fehlerfrei.**

## 3. Avg MARE = 0.81

Durchschnittlicher Rangfehler:
Jede Antwort liegt im Mittel 0.81 Plätze von ihrer wahren Position entfernt.

Wenn die Skala z. B. 1–5 Ränge pro Frage hat, ist <1 schon ziemlich gut.

**Das Modell verfehlt im Schnitt weniger als einen Rangplatz pro Antwort.**

## 4. Avg PairAcc = 0.777

Das ist die Pairwise Accuracy:
Anteil korrekt geordneter Antwortpaare unter allen möglichen Paaren.

0.777 heißt: Rund 78 % der Antwortvergleiche stimmen in der Richtung (besser/schlechter) mit der Wahrheit überein.

**Das ist deutlich über Zufall (0.5), aber es gibt noch rund 22 % falsche Paarvergleiche, die das Modell vertauscht.**

## 5. Total Swaps ↓ = 90 283

Die absolute Zahl der Discordant Pairs, also Paare, die falsch herum sortiert sind, aufsummiert über alle Fragen.

Kein Qualitätsmaß im Verhältnis, aber eine Rohgröße: wie viele „lokale Rangvertauschungen“ das Modell insgesamt macht.

**Je kleiner, desto konsistenter ist die Ordnung.**

## Probleme

1. **Instabile Zentroiden bei kleinen Gruppen:**  
   Wenn eine Notenstufe (z. B. 0.7 oder 0.2) nur sehr wenige Antworten enthält, basiert der Zentroid (Median der Similarities) auf extrem wenig Daten.  
   Dadurch entstehen zufällige oder nicht repräsentative Werte, die das Snapping-Verhalten verzerren.  
   → Beispiel: Eine Note mit nur einer Antwort kann den Medianwert stark verschieben.

2. **Verlust der Monotonie zwischen Notenstufen:**  
   Höhere Noten (z. B. 1.0) sollten im Durchschnitt höhere Similarities haben als niedrigere (z. B. 0.2).  
   Bei kleinen Gruppen oder Ausreißern kann diese monotone Beziehung verloren gehen.  
   → Das führt dazu, dass eine schlechtere Note (0.7) plötzlich einen höheren Zentroidenwert hat als eine bessere (1.0).

3. **Sensitivität gegenüber Ausreißern:**  
   Einzelne extrem hohe oder niedrige Similarities innerhalb einer Note verschieben den Median (oder Mittelwert) spürbar.  
   Besonders problematisch bei Daten mit wenigen Stichproben pro Frage.

4. **Ungleichverteilung der Noten:**  
   Manche Fragen enthalten nur wenige oder gar keine Antworten in bestimmten Notenbereichen.  
   Dadurch fehlen Zentroiden vollständig, was zu inkonsistentem Snapping führt.

---

## Mögliche Lösungsansätze

1. **Fallback für kleine Gruppen:**  
   Falls eine Notenstufe weniger als 3 Antworten hat, ersetze ihren Median durch den Mittelwert der benachbarten Zentroiden.  
   Damit bleibt die Struktur zwischen den Stufen stabil.  
   \[
   c_g =
   \begin{cases}
   \text{median}(\text{sim}_g), & \text{wenn } n_g \ge 3 \\
   \frac{c_{g_\text{prev}} + c_{g_\text{next}}}{2}, & \text{wenn } n_g < 3
   \end{cases}
   \]

2. **Monotone Glättung per Isotone Regression:**  
   Erzwinge eine streng absteigende Reihenfolge der Zentroiden:
   \[
   c_{1.0} \ge c_{0.9} \ge c_{0.8} \ge \dots
   \]
   Implementierbar mit `sklearn.isotonic.IsotonicRegression(increasing=False)`.  
   Dadurch wird sichergestellt, dass höhere Noten immer höhere Similarities behalten.

3. **Shrinkage bei kleinen Gruppen:**  
   Verwende ein „Empirical Bayes“-Verfahren, um den Median kleiner Gruppen zum globalen Durchschnitt hin zu verschieben:  
   \[
   c_g' = \alpha \cdot c_g + (1-\alpha) \cdot \bar{c}, \quad \alpha = \frac{n_g}{n_g + k}
   \]
   (z. B. \(k = 3\))  
   Dadurch wird die Varianz kleiner Gruppen automatisch gedämpft.

4. **Fallback-Zentroiden bei fehlenden Noten:**  
   Wenn für eine bestimmte Note keine Antworten existieren, ersetze ihren Zentroiden durch den der nächsten verfügbaren Note.  
   So bleibt das Mapping vollständig definiert.

5. **Alternative: Clustering anstelle fixer Notenstufen:**  
   Anstatt prozentuale Noten (1.0, 0.8, 0.6, …) zu verwenden, kann man Similarities über alle Antworten mit z. B. *k*-Means in 3–5 semantische Cluster gruppieren.  
   Dadurch entstehen robuste, datengetriebene Bewertungsstufen ohne Abhängigkeit von ungleich verteilten Noten.

