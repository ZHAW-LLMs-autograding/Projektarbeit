import json
import math

INPUT = "/home/sercan/asag_pipeline/pipeline/metriken/benchmarks/vectormodels/char_wb_3_5_min2/predictions_repaired_withoutDuplicates.json"
OUTPUT = "/home/sercan/asag_pipeline/pipeline/metriken/benchmarks/vectormodels/char_wb_3_5_min2/metrics_deduplicate.json"


def load_predictions(path):
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)
    if isinstance(data, dict) and "predictions" in data:
        return data["predictions"]
    return data


def compute_metrics(entries):
    y_true = []
    y_pred = []
    inference_times = []

    for e in entries:
        # skip broken entries (should not happen)
        if e.get("y_pred") is None or e.get("y_true") is None:
            continue
        try:
            yt = float(e["y_true"])
            yp = float(e["y_pred"])
        except:
            continue

        y_true.append(yt)
        y_pred.append(yp)

        if "inference_time_ms" in e:
            inference_times.append(float(e["inference_time_ms"]))

    n = len(y_true)

    # MAE
    mae = sum(abs(yp - yt) for yp, yt in zip(y_pred, y_true)) / n

    # RMSE
    rmse = math.sqrt(sum((yp - yt) ** 2 for yp, yt in zip(y_pred, y_true)) / n)

    # wRMSE (same weights = 1)
    wrmse = rmse

    # total inference time
    total_inference_ms = sum(inference_times)
    total_inference_s = total_inference_ms / 1000.0
    avg_inference_ms = total_inference_ms / n if n > 0 else 0

    return {
        "MAE": mae,
        "RMSE": rmse,
        "wRMSE": wrmse,
        "num_samples_total": len(entries),
        "num_valid_predictions": n,
        "total_inference_time_s": total_inference_s,
        "avg_inference_time_ms": avg_inference_ms
    }


def save_metrics(path, metrics):
    with open(path, "w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2, ensure_ascii=False)


if __name__ == "__main__":
    print(f"🔍 Lade deduplizierte Predictions: {INPUT}")
    entries = load_predictions(INPUT)

    print(f"➡️  Anzahl Einträge: {len(entries)}")

    metrics = compute_metrics(entries)

    print(f"📊 MAE:   {metrics['MAE']}")
    print(f"📊 RMSE:  {metrics['RMSE']}")
    print(f"📊 wRMSE: {metrics['wRMSE']}")
    print(f"⏱  Inferenzzeit total (s): {metrics['total_inference_time_s']}")
    print(f"⏱  Inferenzzeit Ø (ms):   {metrics['avg_inference_time_ms']}")

    print(f"💾 Speichere Metriken nach: {OUTPUT}")
    save_metrics(OUTPUT, metrics)

    print("🎉 Fertig! metrics_deduplicate.json erzeugt.")
