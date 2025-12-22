#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import json
import time
import math
from pathlib import Path

import numpy as np
import pandas as pd
from tqdm import tqdm
from sentence_transformers import SentenceTransformer

# -------------------------------------------------------
# Pfade
# -------------------------------------------------------
ROOT = Path(__file__).resolve().parents[1]

TEST_PARQUET = ROOT / "trainingsdata" / "ASAG2024" / "student_ranking" / "rank_test.parquet"

OUT_DIR = ROOT / "metriken" / "benchmarks" / "sentence_transformer"
OUT_DIR.mkdir(parents=True, exist_ok=True)

PREDICTIONS_PATH = OUT_DIR / "predictions.json"
METRICS_PATH = OUT_DIR / "metrics.json"

# -------------------------------------------------------
# Source-Gewichte für wRMSE (wie bei LLaMA-Base-LoRA)
# -------------------------------------------------------
SOURCE_WEIGHTS = {
    "CSD": 1.0,
    "SAF": 1.0,
    "BeSt": 1.0,
    "Stroud": 1.0,
    "SciEnt": 1.0,
}


def cos_sim(v1, v2):
    """Cosine Similarity."""
    v1 = v1 / (np.linalg.norm(v1) + 1e-12)
    v2 = v2 / (np.linalg.norm(v2) + 1e-12)
    return float((v1 * v2).sum())


def safe_text(x):
    """Sorgt dafür, dass SentenceTransformer immer einen String bekommt."""
    if x is None:
        return ""
    if isinstance(x, float) and math.isnan(x):
        return ""
    return str(x)


def predict_similarity(model, ref, stud):
    """Encode beide Texte und berechne Kosinus-Ähnlichkeit."""
    ref = safe_text(ref)
    stud = safe_text(stud)

    # Falls beide leer sind → Similarity = 0.0
    if ref.strip() == "" and stud.strip() == "":
        return 0.0

    emb1 = model.encode(ref)
    emb2 = model.encode(stud)
    return cos_sim(emb1, emb2)


def main():
    print(f"[INFO] Loading test set: {TEST_PARQUET}")
    df = pd.read_parquet(TEST_PARQUET)
    print(f"[INFO] Loaded {len(df)} samples.")

    print("[INFO] Loading SentenceTransformer: all-MiniLM-L6-v2")
    model = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")

    predictions = []

    total_abs_err = 0.0
    total_sq_err = 0.0
    total_w_sq_err = 0.0
    total_weight = 0.0

    num_valid = 0
    num_total = 0
    total_time_s = 0.0

    print("[INFO] Starting evaluation…")

    for row in tqdm(df.itertuples(index=False), total=len(df)):
        q = safe_text(row.question)
        r = safe_text(row.reference_answer)
        p = safe_text(row.provided_answer)
        y_true = float(row.normalized_grade)
        source = getattr(row, "source", "unknown")

        t0 = time.perf_counter()
        y_pred = predict_similarity(model, r, p)
        t1 = time.perf_counter()

        inf_ms = (t1 - t0) * 1000.0
        total_time_s += (t1 - t0)

        predictions.append({
            "question": q,
            "reference_answer": r,
            "provided_answer": p,
            "y_true": y_true,
            "y_pred": y_pred,
            "source": source,
            "raw_output_text": f"{y_pred:.6f}",
            "inference_time_ms": inf_ms,
        })

        # Fehler / Metriken
        if not (math.isnan(y_pred) or math.isnan(y_true)):
            err = y_pred - y_true
            total_abs_err += abs(err)
            total_sq_err += err * err

            w = SOURCE_WEIGHTS.get(source, 1.0)
            total_w_sq_err += w * err * err
            total_weight += w

            num_valid += 1

        num_total += 1

    # Metriken
    if num_valid > 0:
        mae = total_abs_err / num_valid
        rmse = math.sqrt(total_sq_err / num_valid)
        if total_weight > 0.0:
            w_rmse = math.sqrt(total_w_sq_err / total_weight)
        else:
            w_rmse = float("nan")
        avg_ms = (total_time_s / num_valid) * 1000.0
    else:
        mae = rmse = w_rmse = avg_ms = float("nan")

    metrics = {
        "MAE": mae,
        "RMSE": rmse,
        "wRMSE": w_rmse,
        "num_samples_total": int(num_total),
        "num_valid_predictions": int(num_valid),
        "total_inference_time_s": total_time_s,
        "avg_inference_time_ms": avg_ms,
    }

    with open(PREDICTIONS_PATH, "w", encoding="utf-8") as f:
        json.dump(predictions, f, indent=2, ensure_ascii=False)

    with open(METRICS_PATH, "w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2, ensure_ascii=False)

    print(f"[INFO] Saved predictions → {PREDICTIONS_PATH}")
    print(f"[INFO] Saved metrics     → {METRICS_PATH}")
    print("[INFO] Done.")
    print("[INFO] Metrics:")
    for k, v in metrics.items():
        print(f"  {k}: {v}")


if __name__ == "__main__":
    main()
