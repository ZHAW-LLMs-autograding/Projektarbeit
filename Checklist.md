# ASAG-Pipeline To-Do-Liste (Modellweise)

---

## BERT 2.0

- [ ] **TODO 1 – Datensplit & Setup**

  * Datensplit (Train 80 / Val 10 / Test 10) erstellen
  * Seed = 42 setzen
  * Configs & Checkpoint-Ordner anlegen
  * LoRA / PEFT-Pfad vorbereiten

- [ ] **TODO 2 – Training (Fine-Tuning)**

  * Modell trainieren (ASAG2024, 5-Klassen-Labels)
  * Trainingszeit & GPU-Verbrauch loggen
  * Checkpoints speichern (epochweise)
  * Logs sichern (`train.log`, `wandb.csv`)

- [ ] **TODO 3 – Evaluation**

  * Ladezeit → GPU messen
  * Inferencezeit (einzelne Bewertung) messen
  * Bewertungszeit (10 % Testset) messen
  * Vorhersagen speichern (`preds_bert2.jsonl`)

- [ ] **TODO 4 – Bewertungsmetriken**

  * RMSE, wRMSE, MAE berechnen
  * Längenanalyse (50 / 100 / 200 / max Tokens)
  * wRMSE-Verlauf je Länge plotten

- [ ] **TODO 5 – Ranking & Buckets**

  * Scores → [0, 1] normalisieren
  * Gleichmäßige Bucket-Verteilung (1 / n_Buckets)
  * Kendall τ pro Data_Source berechnen
  * Rangliste & Heatmap erstellen

- [ ] **TODO 6 – Dokumentation**

  * Ergebnisse + Plots in LaTeX / Overleaf einfügen
  * Kurze Analyse (z. B. Einfluss Referenzantwort, Seed-Stabilität)

---

## LLaMA (Zero-Shot Baseline)

- [ ] **TODO 1 – Datensplit & Setup**

  * Prompt-Template definieren („Grade Only“, „Baseline“, „Lenient“)

- [ ] **TODO 2 – Inference**

  * Zero-Shot-Evaluation mit 5 Prompts durchführen
  * Zeiten + Kosten (Tokens) messen
  * Ergebnisse speichern

- [ ] **TODO 3 – Bewertungsmetriken**

  * RMSE / wRMSE berechnen
  * Prompt-Vergleich (Baseline vs Lenient vs Strict)
  * wRMSE vs Cost-Plot erstellen

- [ ] **TODO 4 – Ranking**

  * Scores → [0, 1]
  * Buckets & Kendall τ

- [ ] **TODO 5 – Dokumentation**

  * Prompt-Vergleich diskutieren (Zero- vs Few-Shot-Effekte)

---

## LLaMA (Fine-Tuned on ASAG)

- [ ] **TODO 1 – Setup**

  * Finetune-Konfiguration (LoRA / QLoRA) vorbereiten
  * Split 80 / 10 / 10 + Seed 42

- [ ] **TODO 2 – Training**

  * Fine-Tuning auf ASAG2024
  * Trainingszeit + VRAM loggen

- [ ] **TODO 3 – Evaluation**

  * Benchmark (10 % Testset)
  * Inference-, Bewertungszeit, Ladezeit

- [ ] **TODO 4 – Metriken**

  * RMSE / wRMSE / MAE
  * Längenabhängigkeit

- [ ] **TODO 5 – Ranking**

  * Normalisierung + Buckets + Kendall τ

- [ ] **TODO 6 – Vergleich**

  * Gegen Zero-Shot-LLaMA und GPT-4o Baseline

---

## SentenceTransformer

- [ ] **TODO 1 – Setup**

  * Modell: `all-MiniLM-L6-v2` oder `multi-e5`

- [ ] **TODO 2 – Inference**

  * Cosine Similarity zwischen Student & Reference
  * Ergebnisse speichern

- [ ] **TODO 3 – Metriken**

  * RMSE / wRMSE / MAE
  * Längen- vs. Score-Korrelation

- [ ] **TODO 4 – Ranking**

  * Normalisierung + Buckets + Kendall τ

- [ ] **TODO 5 – Dokumentation**

  * Vergleich zu BERT 2.0 und LLMs darstellen

---

## Vektormodell (Semantic Baseline)

- [ ] **TODO 1 – Setup**

  * Embedding-Modell (`nomic-embed-text-v1`)

- [ ] **TODO 2 – Inference**

  * Cosine Similarity (y = sim( ref, stud ))
  * Ergebnisse speichern

- [ ] **TODO 3 – Metriken**

  * RMSE / wRMSE / MAE

- [ ] **TODO 4 – Ranking**

  * Buckets + Kendall τ

- [ ] **TODO 5 – Dokumentation**

  * Baseline → Vergleich mit anderen Modellen

---

## 🧾 Abschluss

- [ ] Alle Modelle in einer Vergleichstabelle zusammenführen
- [ ] Zeit, Leistung, wRMSE, VRAM visualisieren
- [ ] Finales Ranking (Performance vs Effizienz)
- [ ] Diskussion + Fazit → Overleaf Kapitel “Resultate & Diskussion”

---
