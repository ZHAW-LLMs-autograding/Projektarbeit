# ASAG Evaluationspipeline – Übersicht

---

## Datensetup

* **Datensplit:**
  * 80 % Training / 10 % Validation / 10 % Test
  * Fixe Aufteilung von *Gerome*
  * `RandomSeed = 42`
  * **Shuffle deaktiviert** → gleiche Reihenfolge für alle Modelle

---

## Zeit- und Performanzmetriken

| Metrik             | Beschreibung                                                                    |
| ------------------ | ------------------------------------------------------------------------------- |
| **Trainingszeit**  | Wie lange das Modell für das Training benötigt                                  |
| **Inferencezeit**  | Dauer, um *eine einzelne Bewertung* zu erzeugen                                 |
| **Bewertungszeit** | Zeit für die Bewertung eines kompletten Benchmarks (Testset ≈ 10 % aller Daten) |
| **Ladezeit**       | Zeit, bis das Modell vollständig in die GPU geladen ist                         |
| **GPU VRAM Usage** | Peak-Speichernutzung während Benchmark-Ausführung (`nvidia-smi`)                |

---

## Zeitmessung – Methodik

* **Inferencezeit:**
  * Immer dieselbe Frage + Referenzantwort + Schülerantwort verwenden -> Frage auswählen
  * Ziel: reine Antwortzeit des Modells messen

* **Bewertungszeit:**
  * Selbst erstellter Benchmark (repräsentiert durchschnittliche Klassengrösse) -> Benchmark erstellen aus den Testdaten (10%)
  * 10 % aller Daten = Testset → immer gleich verwenden

---

## Grenzen der Modelle

| Test                               | Ziel                             |
| ---------------------------------- | -------------------------------- |
| 50 Zeichen                         | Minimaler Kontext                |
| 100 Zeichen                        | Kurze Antworten                  |
| 150 Zeichen                        | Mittellang                       |
| 200 Zeichen … bis max Kontextgröße | Obergrenze des Modells ermitteln |

> Ermittelt, wann das Modell in Performance oder Kontextlänge an seine Grenzen stösst.

---

## Bewertungsmetriken

| Kategorie   | Metrik    | Beschreibung                                                 |
| ------------ | --------- | ------------------------------------------------------------ |
| **Fehler**  | RMSE      | Root Mean Squared Error (nicht gewichtet)                    |
|             | wRMSE     | gewichtetes RMSE (pro `data_source`)                         |
| **Ranking** | Kendall τ | Rangordnungskonsistenz zwischen Vorhersagen und echten Noten |

---

## Vergleichende Modelle

| Modell                    | Beschreibung                                                |
| -------------------------- | ----------------------------------------------------------- |
| **Vektormodell**          | Klassisches Feature-Vektor-Baseline (TF-IDF / Bag of Words) |
| **SentenceTransformer**   | Semantisches Embeddingmodell (Cosine Similarity → Note)     |
| **BERT Fine-Tuned**       | Kontextbasiertes Modell, trainiert auf ASAG-Datensatz       |
| **LLaMA (ungefinetuned)** | Zero-Shot / Few-Shot Baseline, nur Prompting                |
| **LLaMA (finetuned)**     | Modell nach LoRA- / PEFT-Training auf ASAG2024              |

---

## Zusammenfassung – Ziel der Pipeline

* Einheitliche **Trainings- und Testbedingungen** für alle Modelle  
* Vergleich über **RMSE / wRMSE / τ**  
* Einheitliches **Bucket-Mapping** pro Frage  
* Klare Messung von **Laufzeit, GPU-Verbrauch und Kontextgrenzen**  
* **Ergebnis:**
  * faire, reproduzierbare und interpretierbare Evaluationsbasis  
  * direkt vergleichbare Modelle unterschiedlicher Architektur

---
