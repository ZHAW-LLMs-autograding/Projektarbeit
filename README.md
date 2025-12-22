# Automatic Short Answer Grading (ASAG) – Projekt-Repository

Dieses Repository enthaelt die **vollstaendige Projektpipeline** fuer die automatische Bewertung von Kurzantworten (*Automatic Short Answer Grading, ASAG*).  
Der Fokus liegt auf **Training, Evaluation, Postprocessing und Cost–Performance-Analyse** unterschiedlicher Modellklassen auf dem **ASAG2024-Datensatz**.

Das Repository dient als **reproduzierbare Experimentbasis** fuer die Projektarbeit.

---

## Projektziele

- Aufbau einer **einheitlichen ASAG-Pipeline**
- Vergleich verschiedener **Modellklassen**:
  - Vektormodelle
  - Encoder-Modelle (BERT, SentenceTransformer)
  - Grosse Sprachmodelle (LLaMA 3.1)
- Evaluation mit **MAE, RMSE und wRMSE**
- Analyse des **Cost–Performance Trade-Offs** (Qualitaet vs. Inferenzzeit)
- Konsistentes **Postprocessing** (Clamping, Ranking)

## Modellklassen

### Vektormodelle

- **TF-IDF** (char_wb 3–5, min_df = 2)
- Sehr geringe Inferenzkosten
- Dient als effiziente Baseline
- Begrenzte semantische Modellierung

### Encoder-Modelle

- **SentenceTransformer (BERT)**
- **BERT finetuned** auf ASAG2024
- Gute Balance zwischen Qualität und Rechenaufwand
- Stark in der Cost–Performance-Analyse

### Grosse Sprachmodelle (LLMs)

- **LLaMA 3.1** (Base / Instruct)
- Optionales **LoRA-Finetuning**
- Deutlich höhere Inferenzkosten
- **GPT-4o** dient als externe Referenz (keine lokale Inferenz)

---

## Evaluation

Folgende Metriken werden berechnet:

- **MAE** (Mean Absolute Error)
- **RMSE** (Root Mean Squared Error)
- **wRMSE** (gewichtete RMSE)

Die finale Modellbewertung erfolgt konsistent über **wRMSE**.  
Zusätzliche Analysen umfassen:

- Clamping auf den Bereich \([0, 1]\)
- Ranking-basierte Auswertungen
- Vergleich von Roh- und Postprocessing-Ergebnissen

---

## Cost–Performance Analyse

- **X-Achse:** mittlere Inferenzzeit pro Antwort (ms)
- **Y-Achse:** \(1 - \mathrm{wRMSE}\) (höher = besser)

**Darstellung:**

- **Kreuze:** evaluierte Modelle
- **Horizontale Linien:** Referenzen (Vektormodell, GPT-4o)

Diese Analyse dient der praktischen Einordnung der Modelle hinsichtlich Effizienz und Qualität.

---

## Reproduzierbarkeit

- Alle Experimente sind deterministisch konfiguriert
- Vorhersagen und Metriken werden persistiert
- Plots und Tabellen werden ausschliesslich aus gespeicherten Ergebnissen erzeugt
- Keine manuelle Nachbearbeitung der Resultate

---

## Installation

```bash
pip install -r requirements.txt

