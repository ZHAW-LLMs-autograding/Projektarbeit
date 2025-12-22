#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Extrahiert alle Trainingsmetriken aus TensorBoard-Logs der BERT-Regressions-
trainings und speichert die Übersicht im üblichen Metriken-Format:
CSV + JSON unter:
    pipeline/metriken/bert/
"""

from tbparse import SummaryReader
import pandas as pd
import json
from pathlib import Path

# -------------------------------------------------------------------
# Pfade
# -------------------------------------------------------------------

ROOT = Path(__file__).resolve().parents[1]

LOG_DIR = ROOT / "artifacts" / "models" / "bert" / "logs"

OUT_DIR = ROOT / "metriken" / "benchmarks" / "bert"
OUT_DIR.mkdir(parents=True, exist_ok=True)

OUT_CSV = OUT_DIR / "training_metrics.csv"
OUT_JSON = OUT_DIR / "training_metrics.json"

# -------------------------------------------------------------------
# Einlesen der TensorBoard Logs
# -------------------------------------------------------------------

print(f"[INFO] Reading BERT training logs from: {LOG_DIR}")

reader = SummaryReader(str(LOG_DIR))
df = reader.scalars  # enthält loss, lr, grad_norm, epoch, step, wall_time

print(f"[INFO] Loaded {len(df)} metric rows")

# -------------------------------------------------------------------
# Speichern als CSV
# -------------------------------------------------------------------

df.to_csv(OUT_CSV, index=False)
print(f"[OK] Saved CSV → {OUT_CSV}")

# -------------------------------------------------------------------
# Speichern als JSON (gruppiert nach 'tag')
# -------------------------------------------------------------------

metrics_by_tag = {}
for tag in df["tag"].unique():
    tag_df = df[df["tag"] == tag]
    metrics_by_tag[tag] = tag_df.to_dict(orient="records")

with open(OUT_JSON, "w", encoding="utf-8") as f:
    json.dump(metrics_by_tag, f, indent=2)

print(f"[OK] Saved JSON → {OUT_JSON}")

print("\n[DONE] BERT training metrics exported successfully.")
