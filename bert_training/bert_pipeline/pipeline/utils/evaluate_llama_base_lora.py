import os
import json
import time
import pandas as pd
from tqdm import tqdm
from inferer_base_lora import LlamaBaseLoraInferer
from evaluator import Evaluator

BASE_MODEL_PATH = "/home/sercan/asag_pipeline/pipeline/artifacts/models/kevin_llama3_base"
LORA_MODEL_PATH = "/home/sercan/asag_pipeline/pipeline/artifacts/models/llama3.1_base_finetuned"
TEST_DATA_PATH = "/home/sercan/asag_pipeline/pipeline/trainingsdata/ASAG2024/student_ranking/rank_test.parquet"

RUN_NAME = "llama_base_lora_run"
OUT_DIR = f"metriken/benchmarks/{RUN_NAME}"
os.makedirs(OUT_DIR, exist_ok=True)

PRED_FILE = f"{OUT_DIR}/predictions.json"
MET_FILE = f"{OUT_DIR}/metrics.json"


def main():
    inferer = LlamaBaseLoraInferer(BASE_MODEL_PATH, LORA_MODEL_PATH)

    df = pd.read_parquet(TEST_DATA_PATH)
    predictions = []

    t0 = time.time()

    for _, row in tqdm(df.iterrows(), total=len(df), desc="Evaluating"):
        q = row["question"]
        r = row["reference_answer"]
        a = row["provided_answer"]
        y_true = float(row["normalized_grade"])
        src = row["data_source"]

        raw, y_pred, dur = inferer.infer(q, r, a)

        predictions.append({
            "question": q,
            "reference_answer": r,
            "provided_answer": a,
            "y_true": y_true,
            "y_pred": y_pred,
            "source": src,
            "raw_output_text": raw,
            "inference_time_ms": dur,
        })

    total_time = time.time() - t0

    with open(PRED_FILE, "w") as f:
        json.dump({"predictions": predictions}, f, indent=2)

    # Metrics
    valid = [p for p in predictions if p["y_pred"] is not None]
    dfv = pd.DataFrame(valid)

    evaluator = Evaluator()

    mae = evaluator.mae(dfv["y_true"], dfv["y_pred"])
    rmse = evaluator.rmse(dfv["y_true"], dfv["y_pred"])
    wrmse = evaluator.wrmse(dfv)

    metrics = {
        "MAE": mae,
        "RMSE": rmse,
        "wRMSE": wrmse,
        "num_samples_total": len(predictions),
        "num_valid_predictions": len(valid),
        "total_inference_time_s": total_time,
        "avg_inference_time_ms": total_time * 1000.0 / len(predictions),
    }

    with open(MET_FILE, "w") as f:
        json.dump(metrics, f, indent=2)

    print("[DONE]")
    print("Predictions:", PRED_FILE)
    print("Metrics:", MET_FILE)


if __name__ == "__main__":
    main()
