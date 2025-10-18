# ADR — Dokumentationsrichtlinien
Hier die erweiterte Liste als Markdown — copy/paste-ready:

## Geschriebenes besprechen (erweitert)

* Jedes Kapitel **soll** Text enthalten — auch kurze, prägnante Einleitungen sind ok.
* Lange Kapitel-Vorbeschreibungen vermeiden; wenn nötig, kurz halten.
* Beschreibungen der Umgebung nur so ausführlich wie nötig (Mehraufwand vermeiden).
* Anhang (Appendix) **ja** — dort zusätzliche Details, Daten, Logs und Abbildungs-Beschreibungen.
* Keine persönlichen Namen im Fließtext; unpersönliche Formulierungen bevorzugen.

### Ergänzungen — Was **erklären** vs. was **weglassen**

* **Erklären**, wenn die Information für den Lesenden **nicht selbstverständlich** ist — z. B.:

  * Konzeptuelles Datensplitting (Train/Val/Test) und warum wir *task-level* splits verwenden.
  * Bewertungsmetriken (wRMSE, MAE, ACC(5)) und was sie messen.
  * Designentscheidungen, die das Ergebnis beeinflussen (z. B. Ordinal vs. Regression, Referenz-Kontext ein/aus).
* **Nicht detailliert erklären** (weglassen), weil ein interessierter Leser das schnell selbst nachschlägt:

  * Allgemeine Begriffe wie „LLM“, „GPU“ oder „Transformer“ — davon geht man aus, dass der Leser Grundwissen hat.
  * Marketing-/Hardware-Spekulationen oder tiefe Hardware-Details (z. B. „L4 mit 24GB und X Tensor-Cores …“) — nur angeben, **wenn** es reproduzierungsrelevant ist.
* **Kurz angeben**, wenn für Reproduzierbarkeit relevant:

  * Konkrete Software-Versionen, Seeds, Konfigurationsdateien, wichtige Hyperparameter (Lernrate, Batchsize, Epochs).
  * Hardware-Angaben **nur** in einfacher Form (z. B. „1× GPU mit ≥24 GB VRAM“), nicht als Werbetext.

### Praktische Empfehlung

* Alles, was nötig ist, damit eine *technisch versierte Person* das Experiment reproduzieren kann, kurz dokumentieren (configs + scripts + seed).
* Alles, was triviales Hintergrundwissen ist, weglassen oder als Fußnote / Link ins Appendix verweisen.
* Halte die Sprache sachlich, knapp und unpersönlich — bei konkreten Arbeitsschritten ist „wir“ erlaubt.

## Kapitel-Richtlinien (kurz)

* In jedem Kapitel mindestens ein bis zwei Sätze, die den Inhalt zusammenfassen.
* Vermeide redundante Vorworte; stattdessen direkt zum Thema kommen.
* Bei konkreten Ausführungen (z. B. Experimente, Befehle) darf **„wir“** verwendet werden.
* Bei Richtlinien/Policy-Texten neutral und passiv formulieren.

## Anhang / Appendix

* Jede Abbildung/Tabelle im Anhang **muss** eine kurze Beschreibung haben: was gezeigt wird und warum es relevant ist.
* Anhänge für Rohdaten, Eval-CSV, Config-Snippets, ausführliche Fehlermeldungen.
* Quellcode-Snippets nur als Referenz; längere Skripte in `supplementary/` verlinken.

## Sprache & Stil

* **Unpersönlich** schreiben — keine Namensnennungen (z. B. „Person X“).
* Direkte ChatGPT-Texte vermeiden; wenn KI-Unterstützung verwendet wurde, neutral kennzeichnen und nachbearbeiten.
* Statische / monotone KI-Formulierungen nachbearbeiten (z. B. mit DeepL rephrasen) für natürlicheren Ton.
* Prägnant, sachlich, technisch — keine ausschweifenden Metaphern.

## Format & Titelblatt

* ZHAW-Format beachten: Jonathan hat das Format vorgegeben — diesem folgen.
* Titelblatt wie in den Vorgaben gestalten (ZHAW-Richtlinie gilt als verbindlich).
* Inhaltliche Formatierung (Kapitelhierarchie, Abbildungsbeschriftung) kann projektintern leicht angepasst werden, aber nicht das Titelblatt-Layout.

---

Wenn du willst, erstelle ich dir die fertigen Markdown-Dateien (`0001-docstyle.md`, `0002-appendix.md`, ...) zum direkten Einchecken.
