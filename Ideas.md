# Projektideen & Erweiterungen

Dieses Dokument sammelt offene Ideen und mögliche Erweiterungen für die Projektarbeit im Kontext **ASAG mit Open-Source-LLMs**.

---

## Modell- und Bewertungsansätze

- **Ordinal Regression / Threshold-based Classification**  
  - Bewertung über stufenweise Schwellenwerte (z. B. 4-Punkte-Skala ja/nein, 3-Punkte-Skala ja/nein).  
  - Ziel: Kontinuierliche Notenverteilung besser abbilden und Klassengrenzen modellieren.

- **Grammatik-Einfluss auf Bewertung**  
  - Untersuchung, wie stark grammatikalische Unterschiede die Bewertung beeinflussen.  
  - Idee: Eingaben automatisch in korrekte Sätze umwandeln (Grammar-Correction) und anschließend bewerten.  
  - Vergleich „mit“ vs. „ohne“ Normalisierung.

---

## Theoretische und methodische Vertiefungen

- **Prompt-Engineering (Kapitel 5.2.4 aus Gérôme-Arbeit)**  
  - Weiterführung der Ansätze, z. B. systematische Variation von Beispielauswahl, Chain-of-Thought-Anweisungen und Bewertungsformaten.  
  - Fokus auf reproduzierbare Prompts und Parameterstabilität.

- **Verteilungsdrift und Modellverhalten über Zeit**  
  - Modelle neigen dazu, bekannte Verteilungen zu approximieren ("Grade Bias").  
  - Mit gezieltem Error-Handling oder Kalibrierung könnte dieses Verhalten kompensiert werden.  
  - Forschungsfrage: Wie stabil bleibt die Bewertung über mehrere Iterationen / Trainingsläufe hinweg?

---

## Neue Anwendungsszenarien

- **ASAG für Unterstufe / Primarstufe**  
  - Vereinfachtes Bewertungs- und Feedback-System.  
  - Fokus: Feedback für Schüler *und* Lehrperson („Fokuslenkung“).  
  - Erste Hypothese: Nicht optimiert, aber eventuell ausreichend robust für einfache Antworten.

- **Grammatik-basierte Feedback-Systeme**  
  - Analyse des Einflusses von sprachlichen Fehlern auf Feedback und Bewertung.  

---

## Weitere Forschungsrichtungen

- **Jailbreak-Experimente**  
  - Untersuchung, wie sich jailbreaked bzw. ungeschützte Modelle in Bewertungsszenarien verhalten.  
  - Relevanz: Sicherheit und Manipulationsresistenz bei Prüfungs-LLMs.

- **Kosten- und Leistungsanalyse**  
  - Vergleich der Kosten pro 1000 Tokens und GPU-Zeit (Inference + Training).  
  - Ziel: Metrik "Kosten pro Güteeinheit" (Cost-Per-Performance-Ratio) zwischen GPU-Grössen und Modellen (Mixtral 8×7B vs GPT-4o etc.).  
  - Ergänzend: Laufzeitprofil (Token/s, Speicherbedarf, Effizienz).

---
