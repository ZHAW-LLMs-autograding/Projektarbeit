# Modelle
* Vektormodell
* SentenceTransformer
* Bert 2.0
* Llama
* Llama finetuned

---

# ✅ ASAG Pipeline – To-Do-Liste (alle Punkte, wie gesagt)

* [ ] **Datensplit erstellen** (Train / Val / Test)
* [ ] **Seed = 42** setzen

---

### 🧮 **Metriken für alle Modelle**

* [ ] RMSE berechnen
* [ ] wRMSE berechnen

---

### ⏱️ **Zeiten messen**

* [ ] Trainingszeit messen
* [ ] Inferencezeit (eine Bewertung) messen
* [ ] Bewertungszeit messen (eigener Benchmark aus 10 % Testdaten)
* [ ] Ladezeit in GPU messen
* [ ] VRAM Usage der Modelle messen

---

### ✂️ **Längenanalyse**

* [ ] Evaluierung pro Grenze

  * [ ] 50 Zeichen
  * [ ] 100 Zeichen
  * [ ] …
  * [ ] bis **maximale Kontextgröße**

---

### 📊 **Ranking & Buckets**

* [ ] Ranking mit Buckets pro **Data_Source** erstellen
* [ ] Kendall τ berechnen
* [ ] Werte des Modells auf 1 zurück normalisieren
* [ ] Antworten gleichmäßig auf Buckets verteilen
* [ ] 1 / n_Buckets berechnen und auf die Bucketwerte runden

---
