#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
ASAG Postprocessing:
- Clamping
- Ranking (raw + clamped)
- Metrics
- Kendall tau (raw + clamped)

Erzeugt pro Modellordner:
    predictions_clamped.json
    predictions_ranking.json
    metrics_clamped.json
    tau.json

Konfiguration:
- Buckets pro data_source kommen aus bucket_stats_overall.csv.
- Rundung: default 3 decimals.
- y_pred_clamped: clamp auf [0,1].
"""

from __future__ import annotations

import argparse
import csv
import json
import math
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Sequence, Tuple


# -----------------------------------------
# Utility functions
# -----------------------------------------

def clamp_0_1(x: float) -> float:
    """Clamp Wert hart auf [0,1]."""
    return max(0.0, min(1.0, x))


def compute_mae_rmse(
    y_true: Sequence[float], y_pred: Sequence[float]
) -> Tuple[float, float]:
    """Berechnet MAE und RMSE."""
    n = len(y_true)
    if n == 0:
        return float("nan"), float("nan")

    abs_err = 0.0
    sq_err = 0.0
    for yt, yp in zip(y_true, y_pred):
        diff = yt - yp
        abs_err += abs(diff)
        sq_err += diff * diff

    mae = abs_err / n
    rmse = math.sqrt(sq_err / n)
    return mae, rmse


def kendall_tau_b(x: Sequence[int], y: Sequence[int]) -> float:
    """
    Kendall's tau_b (tie-aware).
    Implementiert ohne externe Dependencies.
    """
    n = len(x)
    if n != len(y):
        raise ValueError("x und y müssen gleich lang sein.")
    if n < 2:
        return float("nan")

    uniq_x = sorted(set(x))
    uniq_y = sorted(set(y))
    ix = {v: i for i, v in enumerate(uniq_x)}
    iy = {v: i for i, v in enumerate(uniq_y)}
    m_x = len(uniq_x)
    m_y = len(uniq_y)

    # Kontingenzmatrix
    M = [[0]*m_y for _ in range(m_x)]
    for a, b in zip(x, y):
        M[ix[a]][iy[b]] += 1

    row_sums = [sum(row) for row in M]
    col_sums = [sum(M[i][j] for i in range(m_x)) for j in range(m_y)]

    n0 = n * (n - 1) // 2
    n1 = sum(r*(r-1)//2 for r in row_sums)
    n2 = sum(c*(c-1)//2 for c in col_sums)

    # suffix sums
    suffix_ge = [[0]*m_y for _ in range(m_x)]
    suffix_ge_le = [[0]*m_y for _ in range(m_x)]

    # >= matrix
    for i in range(m_x-1, -1, -1):
        for j in range(m_y-1, -1, -1):
            s = M[i][j]
            if i+1 < m_x:
                s += suffix_ge[i+1][j]
            if j+1 < m_y:
                s += suffix_ge[i][j+1]
            if i+1 < m_x and j+1 < m_y:
                s -= suffix_ge[i+1][j+1]
            suffix_ge[i][j] = s

    # >= in x, <= in y
    for i in range(m_x-1, -1, -1):
        for j in range(m_y):
            s = M[i][j]
            if i+1 < m_x:
                s += suffix_ge_le[i+1][j]
            if j-1 >= 0:
                s += suffix_ge_le[i][j-1]
            if i+1 < m_x and j-1 >= 0:
                s -= suffix_ge_le[i+1][j-1]
            suffix_ge_le[i][j] = s

    P, Q = 0, 0
    for i in range(m_x):
        for j in range(m_y):
            nij = M[i][j]
            if nij <= 0:
                continue

            if i+1 < m_x and j+1 < m_y:
                P += nij * suffix_ge[i+1][j+1]

            if i+1 < m_x and j-1 >= 0:
                Q += nij * suffix_ge_le[i+1][j-1]

    denom = math.sqrt((n0 - n1)*(n0 - n2))
    if denom == 0:
        return float("nan")
    return (P - Q) / denom


def load_buckets(csv_path: Path, round_decimals: int) -> Dict[str, List[float]]:
    """
    Lade Buckets pro Quelle aus bucket_stats_overall.csv.
    Rundet Buckets auf round_decimals.
    """
    buckets = defaultdict(set)
    with csv_path.open("r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            src = row["data_source"]
            b = round(float(row["bucket"]), round_decimals)
            buckets[src].add(b)
    return {src: sorted(vals) for src, vals in buckets.items()}


# -----------------------------------------
# Haupt-Logik pro Modellordner
# -----------------------------------------

def process_model(
    model_dir: Path,
    buckets_by_source: Dict[str, List[float]],
    round_decimals: int
):
    print(f"\n=== Verarbeite Modellordner: {model_dir} ===")

    # 1) Predictions laden (raw)
    pred_path = None
    for name in ["predictions_repaired_withoutDuplicates.json", "predictions.json"]:
        p = model_dir / name
        if p.exists():
            pred_path = p
            break

    if pred_path is None:
        print("  -> Keine Predictions-Datei gefunden. Überspringe.")
        return

    with pred_path.open("r", encoding="utf-8") as f:
        predictions_raw = json.load(f)

    if not isinstance(predictions_raw, list) or len(predictions_raw) == 0:
        print("  -> Leere / ungültige Predictions. Überspringe.")
        return

    # 2) predictions_clamped vorbereiten
    predictions_clamped = []
    for e in predictions_raw:
        new_e = dict(e)
        yp = float(e["y_pred"])
        new_e["y_pred"] = clamp_0_1(yp)
        predictions_clamped.append(new_e)

    # 3) metrics_clamped berechnen
    y_true = [float(e["y_true"]) for e in predictions_clamped]
    y_pred = [float(e["y_pred"]) for e in predictions_clamped]

    mae, rmse = compute_mae_rmse(y_true, y_pred)
    metrics = {
        "MAE": mae,
        "RMSE": rmse,
        "wRMSE": rmse,
        "num_samples": len(y_true)
    }

    with (model_dir / "metrics_clamped.json").open("w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=4)

    print("  -> metrics_clamped.json geschrieben.")

    # 4) Ranking TRUE (Buckets → Ranks)
    truth_rank = {}
    for src, buckets in buckets_by_source.items():
        desc = sorted(buckets, reverse=True)
        truth_rank[src] = {b: i+1 for i, b in enumerate(desc)}

    # -----------------------------------------
    # 5) RAW Ranking (predictions_ranking.json)
    # -----------------------------------------
    src_preds_raw = defaultdict(list)
    src_indices_raw = defaultdict(list)

    for idx, e in enumerate(predictions_raw):
        src = e.get("source") or e.get("data_source")
        if src not in buckets_by_source:
            raise ValueError(f"Quelle {src} nicht in bucket_stats_overall.csv")

        v = round(float(e["y_pred"]), round_decimals)
        src_preds_raw[src].append(v)
        src_indices_raw[src].append(idx)

    # Predicted Rank Mapping RAW
    pred_rank_raw = {}
    for src, preds in src_preds_raw.items():
        buckets = buckets_by_source[src]
        union_vals = sorted(set(buckets) | set(preds), reverse=True)
        pred_rank_raw[src] = {v: i+1 for i, v in enumerate(union_vals)}

    predictions_ranking = [dict(e) for e in predictions_raw]

    for src, idxs in src_indices_raw.items():
        tmap = truth_rank[src]
        pmap = pred_rank_raw[src]

        for idx in idxs:
            e = predictions_ranking[idx]
            yt = round(float(e["y_true"]), round_decimals)
            yp = round(float(e["y_pred"]), round_decimals)

            r_true = tmap[yt]
            r_pred = pmap[yp]

            e["ranking_true"] = r_true
            e["ranking_pred"] = r_pred

    with (model_dir / "predictions_ranking.json").open("w", encoding="utf-8") as f:
        json.dump(predictions_ranking, f, indent=4)

    print("  -> predictions_ranking.json geschrieben.")

    # -----------------------------------------
    # 6) CLAMPED Ranking (für tau_clamped)
    # -----------------------------------------
    src_preds_clamped = defaultdict(list)
    src_indices_clamped = defaultdict(list)

    for idx, e in enumerate(predictions_clamped):
        src = e.get("source") or e.get("data_source")
        v = round(float(e["y_pred"]), round_decimals)
        src_preds_clamped[src].append(v)
        src_indices_clamped[src].append(idx)

    pred_rank_clamped = {}
    for src, preds in src_preds_clamped.items():
        buckets = buckets_by_source[src]
        union_vals = sorted(set(buckets) | set(preds), reverse=True)
        pred_rank_clamped[src] = {v: i+1 for i, v in enumerate(union_vals)}

    rank_true_clamped = []
    rank_pred_clamped = []

    for src, idxs in src_indices_clamped.items():
        tmap = truth_rank[src]
        pmap = pred_rank_clamped[src]

        for idx in idxs:
            e = predictions_clamped[idx]
            yt = round(float(e["y_true"]), round_decimals)
            yp = round(float(e["y_pred"]), round_decimals)

            r_true = tmap[yt]
            r_pred = pmap[yp]

            e["ranking_true"] = r_true
            e["ranking_pred"] = r_pred

            rank_true_clamped.append(r_true)
            rank_pred_clamped.append(r_pred)

    # predictions_clamped.json mit Ranking speichern
    with (model_dir / "predictions_ranking_clamped.json").open("w", encoding="utf-8") as f:
        json.dump(predictions_clamped, f, indent=4)

    print("  -> predictions_ranking_clamped.json mit Ranking geschrieben.")

    # -----------------------------------------
    # 7) tau_raw + tau_clamped
    # -----------------------------------------
    rank_true_raw = [e["ranking_true"] for e in predictions_ranking]
    rank_pred_raw = [e["ranking_pred"] for e in predictions_ranking]

    tau_raw = kendall_tau_b(rank_true_raw, rank_pred_raw)
    tau_clamped = kendall_tau_b(rank_true_clamped, rank_pred_clamped)

    tau = {
        "tau_raw": tau_raw,
        "tau_clamped": tau_clamped,
        "num_samples": len(rank_true_raw)
    }

    with (model_dir / "tau.json").open("w", encoding="utf-8") as f:
        json.dump(tau, f, indent=4)

    print(f"  -> tau.json geschrieben.")
    print(f"     tau_raw     = {tau_raw:.6f}")
    print(f"     tau_clamped = {tau_clamped:.6f}")


# -----------------------------------------
# main()
# -----------------------------------------

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--base-dir", type=str, default=".")
    parser.add_argument("--bucket-stats", type=str, required=True)
    parser.add_argument("--round-decimals", type=int, default=3)
    args = parser.parse_args()

    base = Path(args.base_dir).resolve()
    buckets_path = Path(args.bucket_stats).resolve()

    print(f"Basisordner: {base}")
    print(f"Bucket-CSV: {buckets_path}")
    print(f"Rundung: {args.round_decimals} Stellen")

    buckets_by_source = load_buckets(buckets_path, args.round_decimals)

    model_dirs = [
        "bert",
        "llama_base",
        "llama_instruct",
        "llama3.1_base_lora",
        "llama3.1_instruct_lora",
        "sentence_transformer",
        "vectormodels/char_wb_3_5_min2",
    ]

    for md in model_dirs:
        path = base / md
        if not path.exists():
            print(f"  -> Überspringe {path} (existiert nicht)")
            continue

        process_model(path, buckets_by_source, args.round_decimals)


if __name__ == "__main__":
    main()
