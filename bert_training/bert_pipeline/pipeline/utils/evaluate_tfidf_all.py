#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import json
import time
import math
from pathlib import Path

import torch
import pandas as pd
import numpy as np
from tqdm import tqdm
from sklearn.feature_extraction.text import TfidfVectorizer


# -------------------------------------------------------
# Pfade
# -------------------------------------------------------
ROOT = Path(__file__).resolve().parents[1]

TEST_PARQUET = ROOT / "trainingsdata" / "ASAG2024" / "student_ranking" / "rank_test.parquet"

OUT_ROOT = ROOT / "metriken" / "vectormodels"
OUT_ROOT.mkdir(parents=True, exist_ok=True)


# -------------------------------------------------------
# TF-IDF Modelle
# -------------------------------------------------------
CONFIGS = {
    "baseline_unigram": TfidfVectorizer(),
    "unigram_stop": TfidfVectorizer(stop_words='english'),
    "uni_bigram": TfidfVectorizer(ngram_range=(1,2)),
    "uni_bigram_stop": TfidfVectorizer(stop_words='english', ngram_range=(1,2)),
    "unigram_sublinear": TfidfVectorizer(sublinear_tf=True),
    "char_wb_3_5": TfidfVectorizer(analyzer='char_wb', ngram_range=(3,5)),
    "char_wb_3_5_min2": TfidfVectorizer(analyzer='char_wb', ngram_range=(3,5), min_df=2),
    "char_wb_4_6": TfidfVectorizer(analyzer='char_wb', ngram_range=(4,6)),
    "unigram_min2_max0.9": TfidfVectorizer(min_df=2, max_df=0.9),
    "uni_bigram_min2_max0.9": TfidfVectorizer(ngram_range=(1,2), min_df=2, max_df=0.9),
}


# -------------------------------------------------------
# Single prediction: Cosine similarity
# -------------------------------------------------------
def predict_similarity(vec: TfidfVectorizer, ref: str, stud: str):
    texts = [ref, stud]
    X = vec.transform(texts)
    # Dot product works because TF-IDF is L2-normalized
    sim = float(X[0].multiply(X[1]).sum())
    return sim


# -------------------------------------------------------
# Hauptfunktion
# -------------------------------------------------------
def main():
    print(f"[INFO] Loading test set: {TEST_PARQUET}")
    df = pd.read_parquet(TEST_PARQUET)

    # --- Fix missing text fields ---
    text_cols = ["question", "reference_answer", "provided_answer"]
    for col in text_cols:
        if col not in df.columns:
            raise ValueError(f"Missing required column: {col}")
        df[col] = df[col].fillna("").astype(str)

    # Noten
    df["normalized_grade"] = pd.to_numeric(df["normalized_grade"], errors="coerce")
    df = df[df["normalized_grade"].notna()].copy()

    # Source optional
    if "source" not in df.columns:
        df["source"] = "unknown"
    df["source"] = df["source"].fillna("unknown").astype(str)


    required = {"question", "reference_answer", "provided_answer", "normalized_grade"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"Missing columns: {missing}")

    df = df.copy()
    df["source"] = df.get("source", "unknown")

    print(f"[INFO] Loaded {len(df)} samples.")

    # ---------------------------------------------------
    # TF-IDF Modelle durchgehen
    # ---------------------------------------------------
    for name, vec in CONFIGS.items():

        print(f"\n\n=============== {name} ===============")

        OUT_DIR = OUT_ROOT / name
        OUT_DIR.mkdir(parents=True, exist_ok=True)

        PRED_PATH = OUT_DIR / "predictions.json"
        MET_PATH = OUT_DIR / "metrics.json"

        # ---------------------------------------------------
        # Fit Vectorizer auf Ref+Stud-Daten
        # ---------------------------------------------------
        print("[INFO] Fitting TF-IDF vectorizer...")
        corpus = df["reference_answer"].tolist() + df["provided_answer"].tolist()
        vec.fit(corpus)

        predictions = []

        total_abs_err = 0.0
        total_sq_err = 0.0
        num_valid = 0
        num_total = 0

        total_time_s = 0.0

        # ---------------------------------------------------
        # Prediction Loop
        # ---------------------------------------------------
        print("[INFO] Running TF-IDF evaluation...")
        for row in tqdm(df.itertuples(index=False), total=len(df), dynamic_ncols=True):

            question = row.question
            ref = row.reference_answer
            stud = row.provided_answer
            y_true = float(row.normalized_grade)
            source = getattr(row, "source", "unknown")

            t0 = time.perf_counter()
            try:
                y_pred = predict_similarity(vec, ref, stud)
            except Exception as e:
                print(f"[WARN] Prediction failed at sample {num_total}: {e}")
                y_pred = None
                raw = ""
            else:
                raw = f"{y_pred:.6f}"
            t1 = time.perf_counter()

            inf_ms = (t1 - t0) * 1000.0
            total_time_s += (t1 - t0)

            entry = {
                "question": question,
                "reference_answer": ref,
                "provided_answer": stud,
                "y_true": y_true,
                "y_pred": y_pred,
                "source": source,
                "raw_output_text": raw,
                "inference_time_ms": inf_ms,
            }
            predictions.append(entry)

            num_total += 1

            if y_pred is not None and not math.isnan(y_pred):
                num_valid += 1
                err = y_pred - y_true
                total_abs_err += abs(err)
                total_sq_err += err * err

        # ---------------------------------------------------
        # Metriken
        # ---------------------------------------------------
        if num_valid > 0:
            mae = total_abs_err / num_valid
            rmse = math.sqrt(total_sq_err / num_valid)
            avg_ms = (total_time_s / num_valid) * 1000.0
        else:
            mae = rmse = avg_ms = float("nan")

        metrics = {
            "MAE": mae,
            "RMSE": rmse,
            "wRMSE": rmse,  # kein weighting für TF-IDF
            "num_samples_total": int(num_total),
            "num_valid_predictions": int(num_valid),
            "total_inference_time_s": total_time_s,
            "avg_inference_time_ms": avg_ms,
        }

        # ---------------------------------------------------
        # Save JSON
        # ---------------------------------------------------
        with open(PRED_PATH, "w", encoding="utf-8") as f:
            json.dump(predictions, f, ensure_ascii=False, indent=2)

        with open(MET_PATH, "w", encoding="utf-8") as f:
            json.dump(metrics, f, ensure_ascii=False, indent=2)

        print(f"[INFO] Saved predictions to: {PRED_PATH}")
        print(f"[INFO] Saved metrics to: {MET_PATH}")
        print("[INFO] DONE")


if __name__ == "__main__":
    main()
