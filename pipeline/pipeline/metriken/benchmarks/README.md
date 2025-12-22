# Benchmark Übersicht

Dieses Verzeichnis enthält alle Auswertungen der Modelle im ASAG-Projekt.
Für jedes Modell werden drei Hauptdateien erzeugt:

---

## **predictions.json**

Eine Zeile pro Testbeispiel:

```json
{
  "question": "...",
  "reference_answer": "...",
  "provided_answer": "...",
  "y_true": 1.0,
  "y_pred": 0.82,
  "source": "SAF",
  "raw_output_text": "0.82",
  "inference_time_ms": 295.3
}
```

Enthält also **alle Modellvorhersagen**, die wir später für Buckets und Ranglisten verwenden.

---

## **metrics.json**

Gesamtmetriken des Modells:

```json
{
  "MAE": 0.28,
  "RMSE": 0.36,
  "wRMSE": 0.35,
  "num_samples_total": 3851
}
```

**wRMSE** berücksichtigt unterschiedliche Datenquellen (CSD, SAF, BeSt …).

---

## **Trainingsmetriken**

* `training_metrics.csv` oder `.json`
  → Verlauf von Loss, MAE, RMSE während des Trainings
  → Dient zur Kontrolle der Modellkonvergenz.

---

# Postprocessing: Bucketing & Ranking

Nach der Vorhersage erstellen wir pro Frage **Buckets** und **Ranglisten**.
Jede Datenquelle hat eine festgelegte Anzahl Buckets.

Wir verwenden **3 Methoden**:

### **1) Normalisierung auf 0–1 + Buckets**

Modellwerte werden auf [0,1] gemappt und nach **1/n_buckets** eingeteilt.

### **2) Direkte Bucket-Zuordnung**

Modellwerte ohne Normalisierung direkt auf Buckets verteilt.

### **3) Gleichmäßige Rangverteilung**

Top-Antworten kommen in den höchsten Bucket, mittlere in den mittleren usw.
(z. B. bei 15 Antworten: 5–5–5).

---

# Bewertung: Kendall-Tau

Für jede Bucket-Methode vergleichen wir:

* **predicted ranking**
* **true ranking** (aus normalized_grade)

Mit **Kendall-Tau** messen wir:

* τ = 1 → perfekte Reihenfolge
* τ = 0 → zufällig
* τ = –1 → komplett invertiert

Damit bewerten wir, wie gut das Modell sortiert und gruppiert.
