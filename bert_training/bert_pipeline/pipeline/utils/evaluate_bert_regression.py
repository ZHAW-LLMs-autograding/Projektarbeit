import json
import time
import math
from pathlib import Path

import torch
import pandas as pd
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from tqdm import tqdm
import numpy as np

# -------------------------------------------------------
# DATA + MODEL PATHS
# -------------------------------------------------------

ROOT = Path(__file__).resolve().parents[1]

TEST_PATH = ROOT / "trainingsdata" / "ASAG2024" / "student_ranking" / "rank_test.parquet"

BERT_MODEL_PATH = ROOT / "artifacts" / "models" / "bert" / "checkpoint-17500"

OUT_DIR = ROOT / "metriken" / "benchmarks" / "bert"
OUT_DIR.mkdir(parents=True, exist_ok=True)

PRED_FILE = OUT_DIR / "predictions.json"
METRICS_FILE = OUT_DIR / "metrics.json"

# -------------------------------------------------------
# Load Model + Tokenizer
# -------------------------------------------------------

def load_model():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    print(f"[INFO] Loading BERT model: {BERT_MODEL_PATH}")
    tok = AutoTokenizer.from_pretrained(str(BERT_MODEL_PATH), use_fast=True)
    model = AutoModelForSequenceClassification.from_pretrained(
        str(BERT_MODEL_PATH),
    )
    model.to(device)
    model.eval()

    return model, tok, device

# -------------------------------------------------------
# Build prompt (same as BERT training!)
# -------------------------------------------------------

PROMPT_TEMPLATE = (
    "Question:\n{q}\n\n"
    "Reference answer:\n{r}\n\n"
    "Provided answer:\n{p}\n\n"
    "Predict the grade between 0.0 and 1.0:"
)

def build_prompt(q, r, p):
    return PROMPT_TEMPLATE.format(q=q, r=r, p=p)


# -------------------------------------------------------
# Predict single sample
# -------------------------------------------------------

@torch.no_grad()
def predict_single(model, tok, device, q, r, p):
    prompt = build_prompt(q, r, p)

    enc = tok(
        prompt,
        truncation=True,
        max_length=256,
        return_tensors="pt",
    ).to(device)

    out = model(**enc)
    pred = float(out.logits.view(-1).item())

    # Clip to [0,1] for consistency
    pred = max(0.0, min(1.0, pred))

    return pred, prompt


# -------------------------------------------------------
# Main Evaluation Loop
# -------------------------------------------------------

def main():
    print(f"[INFO] Loading test set: {TEST_PATH}")
    df = pd.read_parquet(TEST_PATH)

    # Clean inputs
    df["question"] = df["question"].fillna("").astype(str)
    df["reference_answer"] = df["reference_answer"].fillna("").astype(str)
    df["provided_answer"] = df["provided_answer"].fillna("").astype(str)
    df["normalized_grade"] = df["normalized_grade"].astype(float)

    model, tok, device = load_model()

    predictions = []
    num_total = 0
    num_valid = 0
    total_time = 0.0

    total_abs_err = 0.0
    total_sq_err = 0.0

    print("[INFO] Starting evaluation…")

    for row in tqdm(df.itertuples(index=False), total=len(df), dynamic_ncols=True):
        q = row.question
        r = row.reference_answer
        p = row.provided_answer
        y_true = float(row.normalized_grade)
        source = getattr(row, "source", "unknown")

        t0 = time.perf_counter()

        try:
            y_pred, raw_prompt = predict_single(model, tok, device, q, r, p)
        except Exception as e:
            print(f"[WARN] Prediction failed at index {num_total}: {e}")
            y_pred = None
            raw_text = ""
        else:
            raw_text = f"{y_pred:.6f}"

        t1 = time.perf_counter()
        inf_ms = (t1 - t0) * 1000.0
        total_time += (t1 - t0)

        entry = {
            "question": q,
            "reference_answer": r,
            "provided_answer": p,
            "y_true": y_true,
            "y_pred": y_pred,
            "source": source,
            "raw_output_text": raw_text,
            "inference_time_ms": inf_ms,
        }
        predictions.append(entry)

        num_total += 1

        if y_pred is not None and not math.isnan(y_pred):
            num_valid += 1
            err = y_pred - y_true
            total_abs_err += abs(err)
            total_sq_err += err * err

    # -------------------------
    # METRICS
    # -------------------------

    if num_valid > 0:
        mae = total_abs_err / num_valid
        rmse = math.sqrt(total_sq_err / num_valid)
        avg_ms = (total_time / num_valid) * 1000.0
    else:
        mae = rmse = avg_ms = float("nan")

    metrics = {
        "MAE": mae,
        "RMSE": rmse,
        "wRMSE": rmse,  # Weighting irrelevant here
        "num_samples_total": num_total,
        "num_valid_predictions": num_valid,
        "total_inference_time_s": total_time,
        "avg_inference_time_ms": avg_ms,
    }

    # -------------------------
    # SAVE OUTPUT
    # -------------------------

    with open(PRED_FILE, "w", encoding="utf-8") as f:
        json.dump(predictions, f, indent=2, ensure_ascii=False)

    with open(METRICS_FILE, "w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2, ensure_ascii=False)

    print("\n[INFO] Evaluation complete!")
    print(f"- Predictions → {PRED_FILE}")
    print(f"- Metrics     → {METRICS_FILE}")

    print("\n[INFO] Metrics:")
    for k, v in metrics.items():
        print(f"  {k}: {v}")


if __name__ == "__main__":
    main()
