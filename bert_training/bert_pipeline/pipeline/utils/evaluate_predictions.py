import json
import math
import pandas as pd
import sys
from pathlib import Path


def evaluate(prediction_file, label_file):
    """
    prediction_file: Pfad zu predictions.json
    label_file: original ASAG2024 test set (rank_test.parquet)
    """

    # Predictions laden
    with open(prediction_file, "r") as f:
        preds = json.load(f)

    # Labels + Weights laden
    df = pd.read_parquet(label_file)

    # Wir benötigen die Originalgewichte, da wRMSE sie benutzt
    weights = df["weight"].to_numpy()

    # Aus Pred.json holen wir y_true und y_pred (in gleicher Reihenfolge)
    y_true = [p["y_true"] for p in preds]
    y_pred = [p["y_pred"] for p in preds]

    # MAE
    mae = sum(abs(p - t) for p, t in zip(y_pred, y_true)) / len(y_true)

    # RMSE
    mse = sum((p - t) ** 2 for p, t in zip(y_pred, y_true)) / len(y_true)
    rmse = math.sqrt(mse)

    # wRMSE
    # gewichtetes Quadrat (Fehler^2 * Gewicht)
    w_mse = sum(weights[i] * ((y_pred[i] - y_true[i]) ** 2)
                for i in range(len(y_true)))
    wrmse = math.sqrt(w_mse)

    return {
        "MAE": mae,
        "RMSE": rmse,
        "wRMSE": wrmse,
        "predictions": preds,
    }


if __name__ == "__main__":
    if len(sys.argv) != 3:
        print("Usage: python evaluate_predictions.py <predictions.json> <rank_test.parquet>")
        sys.exit(1)

    pred_file = Path(sys.argv[1])
    label_file = Path(sys.argv[2])

    out = evaluate(pred_file, label_file)

    # Neues File speichern
    out_file = pred_file.parent / "evaluation.json"
    with open(out_file, "w") as f:
        json.dump(out, f, indent=4)

    print(f"[OK] Evaluation gespeichert unter: {out_file}")
    print(json.dumps(out, indent=4))

