#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, Optional, Tuple, List

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


# -----------------------------
# Pfade
# -----------------------------
ROOT = Path(__file__).resolve().parent
PLOTS_DIR = ROOT / "plots"
PLOTS_DIR.mkdir(parents=True, exist_ok=True)

# -----------------------------
# Modelle (Reihenfolge im Plot)
# -----------------------------
MODELS = [
    ("TF-IDF (char_wb_3_5_min2)", Path("vectormodels/char_wb_3_5_min2")),
    ("SentenceTransformer BERT", Path("sentence_transformer")),
    ("BERT finetuned", Path("bert")),
    ("LLaMA 8B base", Path("llama_base")),
    ("LLaMA 8B instruct", Path("llama_instruct")),
    ("LLaMA 8B base (LoRA FT)", Path("llama3.1_base_lora")),
    ("LLaMA 8B instruct (LoRA FT)", Path("llama3.1_instruct_lora")),
]

# -----------------------------
# Parameteranzahl
# -----------------------------
PARAMS: Dict[str, Optional[float]] = {
    "TF-IDF (char_wb_3_5_min2)": None,
    "SentenceTransformer BERT": 110e6,
    "BERT finetuned": 110e6,
    "LLaMA 8B base": 8_000e6,
    "LLaMA 8B instruct": 8_000e6,
    "LLaMA 8B base (LoRA FT)": 8_000e6,
    "LLaMA 8B instruct (LoRA FT)": 8_000e6,
}

# -----------------------------
# VRAM Fallback-Ranges (GB)
# -----------------------------
VRAM_INF_GB_FALLBACK: Dict[str, Optional[Tuple[float, float]]] = {
    "TF-IDF (char_wb_3_5_min2)": (0.1, 1.0),
    "SentenceTransformer BERT": (1.0, 2.0),
    "BERT finetuned": (1.0, 2.0),
    "LLaMA 8B base": (12.0, 18.0),
    "LLaMA 8B instruct": (12.0, 18.0),
    "LLaMA 8B base (LoRA FT)": (12.0, 18.0),
    "LLaMA 8B instruct (LoRA FT)": (12.0, 18.0),
}

VRAM_TRAIN_GB_FALLBACK: Dict[str, Optional[Tuple[float, float]]] = {
    "TF-IDF (char_wb_3_5_min2)": None,
    "SentenceTransformer BERT": None,
    "BERT finetuned": (6.0, 8.0),
    "LLaMA 8B base": None,
    "LLaMA 8B instruct": None,
    "LLaMA 8B base (LoRA FT)": (20.0, 24.0),
    "LLaMA 8B instruct (LoRA FT)": (20.0, 24.0),
}

# -----------------------------
# Trainingslogs (nur LoRA FT)
# -----------------------------
TRAIN_LOGS_CSV: Dict[str, Path] = {
    "LLaMA 8B base (LoRA FT)": Path("trainer_logs_base.csv"),
    "LLaMA 8B instruct (LoRA FT)": Path("trainer_logs_instruct.csv"),
}

VRAM_COL_CANDIDATES = [
    "vram_peak_gb",
    "vram_reserved_gb",
    "gpu_peak_gb",
    "gpu_reserved_gb",
    "max_memory_allocated_gb",
    "max_memory_reserved_gb",
]

PREDICTION_CANDIDATES = [
    "predictions_repaired_withoutDuplicates.json",
    "predictions_with_ranking.json",
    "predictions_clamped.json",
    "predictions.json",
]

# -----------------------------
# Helper
# -----------------------------
def find_predictions_json(model_dir: Path) -> Optional[Path]:
    for name in PREDICTION_CANDIDATES:
        p = model_dir / name
        if p.exists():
            return p
    hits = sorted(model_dir.glob("predictions*.json"))
    return hits[0] if hits else None


def load_inference_time_ms(pred_path: Path) -> Optional[np.ndarray]:
    try:
        data = json.loads(pred_path.read_text(encoding="utf-8"))
        vals = [
            float(r["inference_time_ms"])
            for r in data
            if isinstance(r, dict) and "inference_time_ms" in r and r["inference_time_ms"] is not None
        ]
        return np.array(vals, dtype=float) if vals else None
    except Exception:
        return None


def mid_err(r: Optional[Tuple[float, float]]) -> Tuple[float, float]:
    if r is None:
        return (np.nan, np.nan)
    lo, hi = r
    return (0.5 * (lo + hi), 0.5 * (hi - lo))


def epoch_time_from_csv_timestamp(csv_path: Path) -> Optional[float]:
    """
    Epoch-Dauer = Mittelwert der Zeitdifferenzen zwischen aufeinanderfolgenden
    Epoch-Timestamps (bei dir: Epoch-Ende nach Evaluation).
    Erkennt Unix-Sekunden vs. Unix-Millisekunden robust.
    """
    if not csv_path.exists():
        return None

    df = pd.read_csv(csv_path)
    colmap = {c.lower(): c for c in df.columns}
    if "epoch" not in colmap or "timestamp" not in colmap:
        return None

    tmp = pd.DataFrame({
        "epoch": pd.to_numeric(df[colmap["epoch"]], errors="coerce"),
        "ts": pd.to_numeric(df[colmap["timestamp"]], errors="coerce"),
    }).dropna()

    if len(tmp) < 2:
        return None

    per_epoch = tmp.groupby("epoch", as_index=False)["ts"].min().sort_values("epoch")
    diffs = per_epoch["ts"].diff().dropna().astype(float)
    mean_diff = float(diffs.mean())

    # Einheit erkennen
    if float(per_epoch["ts"].median()) > 1e12:  # Unix ms
        mean_diff /= 1000.0

    return mean_diff


def vram_range_from_training_csv(csv_path: Path) -> Optional[Tuple[float, float]]:
    if not csv_path.exists():
        return None

    df = pd.read_csv(csv_path)
    colmap = {c.lower(): c for c in df.columns}

    for cand in VRAM_COL_CANDIDATES:
        if cand.lower() in colmap:
            s = pd.to_numeric(df[colmap[cand.lower()]], errors="coerce").dropna()
            if not s.empty:
                lo, hi = float(s.min()), float(s.max())
                if hi > 1000:  # vermutlich MB
                    lo /= 1024.0
                    hi /= 1024.0
                return (lo, hi)
    return None


def range_to_arrays(labels, ranges):
    mids, errs = [], []
    for l in labels:
        m, e = mid_err(ranges.get(l))
        mids.append(m)
        errs.append(e)
    return np.array(mids), np.array(errs)


# -----------------------------
# Daten sammeln
# -----------------------------
labels = [m[0] for m in MODELS]
dirs = [ROOT / m[1] for m in MODELS]
x = np.arange(len(labels))

# Inferenzzeiten
infer_mid_s, infer_err_s = [], []
for lab, d in zip(labels, dirs):
    p = find_predictions_json(d)
    arr = load_inference_time_ms(p) if p else None
    if arr is None:
        infer_mid_s.append(np.nan)
        infer_err_s.append(np.nan)
    else:
        infer_mid_s.append(arr.mean() / 1000.0)
        infer_err_s.append((arr.max() - arr.min()) / 2 / 1000.0)

# Training/Epoch
train_mid_s, train_err_s = [], []
for lab in labels:
    if lab in TRAIN_LOGS_CSV:
        t = epoch_time_from_csv_timestamp(ROOT / TRAIN_LOGS_CSV[lab])
        train_mid_s.append(t)
        train_err_s.append(0.0)
    else:
        train_mid_s.append(np.nan)
        train_err_s.append(np.nan)

# VRAM
vram_train = dict(VRAM_TRAIN_GB_FALLBACK)
for lab, rel in TRAIN_LOGS_CSV.items():
    m = vram_range_from_training_csv(ROOT / rel)
    if m:
        vram_train[lab] = m

vram_inf = dict(VRAM_INF_GB_FALLBACK)

# -----------------------------
# Plot 1: Parameter
# -----------------------------
params = np.array([PARAMS.get(l, np.nan) for l in labels], dtype=float)
mask = ~np.isnan(params)

fig, ax = plt.subplots(figsize=(12, 4))
ax.bar(x[mask], params[mask])
ax.set_yscale("log")
ax.set_xticks(x, labels, rotation=20, ha="right")
ax.set_ylabel("Parameter (log)")
ax.set_title("Parameteranzahl pro Modell")
ax.grid(True, which="both", axis="y", linestyle="--", alpha=0.35)
fig.tight_layout()
fig.savefig(PLOTS_DIR / "resource_01_params.png", dpi=220)
plt.close(fig)

# -----------------------------
# Plot 2: VRAM
# -----------------------------
inf_mid, inf_err = range_to_arrays(labels, vram_inf)
trn_mid, trn_err = range_to_arrays(labels, vram_train)

w = 0.38
fig, ax = plt.subplots(figsize=(12, 4))
ax.bar(x - w/2, inf_mid, w, yerr=inf_err, capsize=4, label="VRAM Inferenz")
ax.bar(x + w/2, trn_mid, w, yerr=trn_err, capsize=4, label="VRAM Training")
ax.set_xticks(x, labels, rotation=20, ha="right")
ax.set_ylabel("VRAM (GB)")
ax.set_title("VRAM-Bedarf")
ax.set_yticks([0, 1, 2, 4, 6, 8, 12, 16, 24, 32])
ax.grid(True, axis="y", linestyle="--", alpha=0.35)
ax.legend(loc="upper left")
fig.tight_layout()
fig.savefig(PLOTS_DIR / "resource_02_vram.png", dpi=220)
plt.close(fig)

# -----------------------------
# Plot 3: Zeiten (sek / min / h)
# -----------------------------
fig, ax = plt.subplots(figsize=(12, 4))
ax.bar(x - w/2, train_mid_s, w, label="Train / Epoch")
ax.bar(x + w/2, infer_mid_s, w, label="Inferenz / Antwort")

ax.set_yscale("log")
ax.set_xticks(x, labels, rotation=20, ha="right")
ax.set_yticks([1e-3, 1e-2, 1e-1, 1, 60, 600, 3600, 21600])
ax.set_yticklabels([
    "1 ms", "10 ms", "100 ms",
    "1 s", "1 min", "10 min",
    "1 h", "6 h"
])
ax.set_ylabel("Zeit (log)")
ax.set_title("Zeiten: Training vs. Inferenz")
ax.grid(True, which="both", axis="y", linestyle="--", alpha=0.35)
ax.legend(loc="upper left")
fig.tight_layout()
fig.savefig(PLOTS_DIR / "resource_03_times.png", dpi=220)
plt.close(fig)

print("Plots erstellt.")
