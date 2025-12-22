import os
import json
import time
import pandas as pd
from tqdm import tqdm
from inferer_instruct import LlamaInstructInferer
from evaluator import Evaluator

MODEL_PATH = "artifacts/models/huggingface__meta-llama__Meta-Llama-3.1-8B-Instruct"
TEST_DATA_PATH = "trainingsdata/ASAG2024/student_ranking/rank_test.parquet"

RUN_NAME = "llama_instruct_run"
OUT_DIR = f"metriken/benchmarks/{RUN_NAME}"
os.makedirs(OUT_DIR, exist_ok=True)

PREDICTIONS_FILE = f"{OUT_DIR}/predictions.json"
METRICS_FILE = f"{OUT_DIR}/metrics.json"


def main():
    inferer = LlamaInstructInferer(MODEL_PATH)
    df = pd.read_parquet(TEST_DATA_PATH)

    predictions = []
    start_eval = time.time()

    for _, row in tqdm(df.iterrows(), total=len(df), desc="Evaluating"):
        q = row["question"]
        r = row["reference_answer"]
        a = row["provided_answer"]
        y_true = row["normalized_grade"]
        source = row["data_source"]

        raw_text, y_pred, infer_time_ms = inferer.infer(q, r, a)

        predictions.append({
            "question": q,
            "reference_answer": r,
            "provided_answer": a,
            "y_true": float(y_true),
            "y_pred": float(y_pred) if y_pred is not None else None,
            "source": source,
            "raw_output_text": raw_text,
            "inference_time_ms": infer_time_ms,
        })

    total_eval_time = time.time() - start_eval

    with open(PREDICTIONS_FILE, "w") as f:
        json.dump({"predictions": predictions}, f, indent=2)

    evaluator = Evaluator()
    valid = [p for p in predictions if p["y_pred"] is not None]
    df_valid = pd.DataFrame(valid)

    mae = evaluator.mae(df_valid["y_true"], df_valid["y_pred"])
    rmse = evaluator.rmse(df_valid["y_true"], df_valid["y_pred"])
    wrmse = evaluator.wrmse(df_valid)

    metrics = {
        "MAE": mae,
        "RMSE": rmse,
        "wRMSE": wrmse,
        "num_samples_total": len(predictions),
        "num_valid_predictions": len(valid),
        "total_inference_time_s": total_eval_time,
        "avg_inference_time_ms": total_eval_time * 1000.0 / len(predictions),
    }

    with open(METRICS_FILE, "w") as f:
        json.dump(metrics, f, indent=2)

    print("[DONE]")


if __name__ == "__main__":
    main()
