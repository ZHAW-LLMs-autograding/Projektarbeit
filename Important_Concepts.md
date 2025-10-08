# Wichtige Begriffe – ASAG & Modelltraining

---

## Grundbegriffe

### **Transformer**
Ein neuronales Netzwerk, das Texte versteht, indem es die Beziehungen zwischen Wörtern analysiert – das Grundprinzip hinter modernen Sprachmodellen.

### **BERT (Bidirectional Encoder Representations from Transformers)**
Ein Sprachmodell, das **Wörter im Kontext** versteht – es liest Sätze von links **und** rechts gleichzeitig, um Bedeutungen besser zu erkennen.

### **MLM (Masked Language Modeling)**
Trainingsmethode bei BERT:  
Einige Wörter werden "maskiert" (`[MASK]`), und das Modell muss sie erraten → hilft beim Sprachverständnis.

---

## Architektur-Komponenten

### **Encoder**
Teil des Modells, der den Text "versteht" – also Eingaben in **numerische Repräsentationen (Vektoren)** umwandelt.

### **MLM-Head**
Letzte Schicht von BERT, die die Vorhersage macht.  
Besteht aus:
- **Projektion:** Lineare Abbildung der versteckten Zustände auf das Vokabular (alle Wörter).  
- **Bias-Vektor:** Eine kleine Korrektur (Offset) pro Token, um Vorhersagen zu verfeinern.

---

## Trainingsmethoden

### **Einfrieren (Freeze)**
Teile des Modells (z. B. Encoder) werden **nicht mehr verändert**, um Rechenzeit zu sparen oder nur bestimmte Schichten zu trainieren.

### **Bias-only Training**
Nur der Bias-Vektor wird trainiert.  
-> Sehr leichtgewichtig, aber **versteht den Kontext kaum**. Gleicht dem "Hardcoden" von Ergebnissen. 

### **Head-only Training**
Nur die letzte Schicht (MLM-Head) wird trainiert.  
-> Lernt **kontextspezifisch**, aber bleibt auf die Ausgabeebene beschränkt.

### **LoRA (Low-Rank Adaptation)**
Erweiterung, die nur **kleine Zusatzmatrizen** trainiert, statt das ganze Modell.  
-> Spart Speicher und Zeit, bleibt aber sehr effektiv.

### **QLoRA**
Kombination von **LoRA + Quantisierung (4-bit)**.  
-> Grosse Modelle können auf **kleinen GPUs** trainiert werden.

#### Vergleich
| Methode        | Rechenaufwand | Kontextverständnis | Beschreibung |
|----------------|----------------|---------------------|---------------|
| **Bias-only**  | sehr gering    | ❌ kaum             | Nur Bias anpassen |
| **Head-only**  | gering         | ✅ etwas            | Nur letzte Schicht |
| **LoRA/QLoRA** | effizient      | ✅✅ gut             | Beste Balance |

---

## Relevanz für ASAG

**ASAG (Automatic Short Answer Grading)** = automatische Bewertung kurzer Freitextantworten.

- Modell muss **Kontext verstehen**.
- **Bias-only:** zu simpel.
- **Head-only:** brauchbar, aber begrenzt.
- **LoRA/QLoRA:** optimal für **leistungsfähige, aber ressourcenschonende Modelle**.

---

## Weitere Begriffe

### **Overfitting**
Modell lernt Trainingsdaten **zu gut auswendig** → funktioniert schlecht auf neuen Daten.

### **Quantisierung**
Reduziert Zahlenpräzision in den Neuronen (z. B. 16-bit → 4-bit), um **Speicher und Rechenleistung zu sparen**.

### **Adam**
Ist ein weit verbreiteter Optimierer für neuronale Netze, der die Lernrate für jedes Gewicht automatisch anpasst.
Er kombiniert die Ideen von **Momentum** (merkt sich vergangene Gradientenrichtungen) und **Adaptive Learning Rate** (passt die Schrittgrösse je Parameter an).
Dadurch lernt das Modell **schneller und stabiler** als mit herkömmlichem Gradientenabstieg, allerdings **ohne korrektes Weight Decay**, was zu leichtem Overfitting führen kann.

### **AdamW**
Ein moderner Optimierer, der die Lernschritte eines Modells anpasst, um den Fehler möglichst effizient zu verringern.
Er verbessert den klassischen Adam, indem er Weight Decay korrekt implementiert – das bedeutet, grosse Gewichte werden leicht abgeschwächt, um Overfitting zu vermeiden.
Dadurch trainieren Modelle stabiler, verallgemeinern besser und konvergieren schneller.

---
