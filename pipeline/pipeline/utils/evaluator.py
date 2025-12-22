import json
import math
import numpy as np
from scipy.stats import kendalltau
import os
import pandas as pd


class Evaluator:
    """
    Evaluator für Predictions des Inferers.
    Speichert Ergebnisse automatisch nach:
        metriken/evals/<run_name>/eval_results.json
    """

    def __init__(self, run_name="default_run"):
        self.output_dir = os.path.join("metriken", "evals", run_name)
        os.makedirs(self.output_dir, exist_ok=True)

    # ---------------------------------------------------------------
    # RMSE
    # ---------------------------------------------------------------
    def rmse(self, y_true, y_pred):
        mse = np.mean((y_true - y_pred) ** 2)
        return float(math.sqrt(mse))

    # ---------------------------------------------------------------
    # MAE
    # ---------------------------------------------------------------
    def mae(self, y_true, y_pred):
        return float(np.mean(np.abs(y_true - y_pred)))

    # ---------------------------------------------------------------
    # wRMSE pro Quelle
    # ---------------------------------------------------------------
    def wrmse(self, df):
        sources = df["source"].unique()
        total_weight = 0.0
        weighted_error = 0.0

        for src in sources:
            subset = df[df["source"] == src]
            if len(subset) == 0:
                continue

            rmse_s = self.rmse(subset["y_true"], subset["y_pred"])
            weight_s = len(subset)

            weighted_error += rmse_s * weight_s
            total_weight += weight_s

        if total_weight == 0:
            return 0.0

        return float(weighted_error / total_weight)

    # ---------------------------------------------------------------
    # Kendall τ
    # ---------------------------------------------------------------
    def kendall(self, y_true, y_pred):
        try:
            tau, _ = kendalltau(y_true, y_pred)
            return float(tau)
        except Exception:
            return 0.0

    # ---------------------------------------------------------------
    # Buckets
    # ---------------------------------------------------------------
    def buckets_1n(self, y_pred, n=5):
        step = 1.0 / n
        return [int(min(p / step, n - 1)) for p in y_pred]

    def buckets_normalized(self, y_pred):
        y = np.array(y_pred)
        min_v, max_v = np.min(y), np.max(y)
        if max_v == min_v:
            return [0 for _ in y_pred]
        norm = (y - min_v) / (max_v - min_v)
        return norm.tolist()

    def buckets_equal(self, y_pred, n=5):
        df = pd.DataFrame({"pred": y_pred})
        df["bucket"] = pd.qcut(df["pred"], q=n, labels=False, duplicates="drop")
        return df["bucket"].fillna(0).astype(int).tolist()

    # ---------------------------------------------------------------
    # Hauptauswertung
    # ---------------------------------------------------------------
    def evaluate(self, predictions, runtime_data=None):
        df = pd.DataFrame(predictions)

        y_true = df["y_true"].values
        y_pred = df["y_pred"].values

        results = {
            "rmse": self.rmse(y_true, y_pred),
            "mae": self.mae(y_true, y_pred),
            "wrmse": self.wrmse(df),
            "kendall_tau": self.kendall(y_true, y_pred),

            "buckets_1n": self.buckets_1n(y_pred),
            "buckets_normalized": self.buckets_normalized(y_pred),
            "buckets_equal": self.buckets_equal(y_pred),

            "runtime": runtime_data if runtime_data else {}
        }

        # Ergebnisse speichern
        out_path = os.path.join(self.output_dir, "eval_results.json")
        with open(out_path, "w") as f:
            json.dump(results, f, indent=4)

        print(f"[OK] Evaluation gespeichert unter: {out_path}")
        return results
