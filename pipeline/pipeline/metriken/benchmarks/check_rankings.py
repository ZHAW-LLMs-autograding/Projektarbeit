#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Check-Script fuer predictions_ranking.json

- Laedt bucket_stats_overall.csv, um Buckets pro data_source zu bekommen
- Laedt predictions_ranking.json
- Rekonstruiert fuer jede Quelle (source/data_source):
    * erwartetes ranking_true (nur aus Buckets, absteigend)
    * erwartetes ranking_pred (Buckets ∪ alle y_pred dieser Quelle, absteigend)
- Vergleicht erwartete Ranks mit den im JSON gespeicherten ranking_true / ranking_pred
- Gibt eine kurze Zusammenfassung und einige Beispiele von Abweichungen aus.
"""

from __future__ import annotations

import argparse
import csv
import json
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Tuple


def load_buckets_by_source(csv_path: Path, round_decimals: int = 5) -> Dict[str, List[float]]:
    """
    Lade Buckets pro data_source aus bucket_stats_overall.csv.

    Erwartete Spalten: data_source, bucket, count
    count wird ignoriert, bucket gerundet.
    """
    buckets_by_src: Dict[str, set] = defaultdict(set)
    with csv_path.open("r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            src = row["data_source"]
            b = round(float(row["bucket"]), round_decimals)
            buckets_by_src[src].add(b)
    return {src: sorted(vals) for src, vals in buckets_by_src.items()}


def build_rank_maps_for_source(
    buckets: List[float],
    y_preds: List[float],
) -> Tuple[Dict[float, int], Dict[float, int]]:
    """
    Erzeuge:

    - truth_rank_map: Rank pro Bucket (absteigend, 1 = beste Note)
    - pred_rank_map: Rank pro Wert in union(Buckets, y_preds) (absteigend)
    """
    # Groundtruth: nur Buckets
    sorted_desc_buckets = sorted(buckets, reverse=True)
    truth_rank_map = {b: rank + 1 for rank, b in enumerate(sorted_desc_buckets)}

    # Pred: Union aus Buckets und allen y_pred
    union_vals = set(buckets) | set(y_preds)
    sorted_desc_union = sorted(union_vals, reverse=True)
    pred_rank_map = {v: rank + 1 for rank, v in enumerate(sorted_desc_union)}

    return truth_rank_map, pred_rank_map


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Checke konsistenz von ranking_true / ranking_pred in predictions_ranking.json"
    )
    parser.add_argument(
        "--pred-file",
        type=str,
        required=True,
        help="Pfad zu predictions_ranking.json",
    )
    parser.add_argument(
        "--bucket-stats",
        type=str,
        required=True,
        help="Pfad zu bucket_stats_overall.csv",
    )
    parser.add_argument(
        "--round-decimals",
        type=int,
        default=3,
        help="Nachkommastellen fuer Rundung von Noten (Default: 5).",
    )
    parser.add_argument(
        "--max-print",
        type=int,
        default=20,
        help="Maximale Anzahl von Beispielen, die bei Abweichungen ausgegeben werden.",
    )
    args = parser.parse_args()

    pred_path = Path(args.pred_file).resolve()
    bucket_stats_path = Path(args.bucket_stats).resolve()
    round_decimals = args.round_decimals
    max_print = args.max_print

    print(f"Predictions-Datei: {pred_path}")
    print(f"Bucket-Stats CSV:  {bucket_stats_path}")

    if not pred_path.exists():
        raise FileNotFoundError(f"predictions_ranking.json nicht gefunden: {pred_path}")
    if not bucket_stats_path.exists():
        raise FileNotFoundError(f"bucket_stats_overall.csv nicht gefunden: {bucket_stats_path}")

    # Buckets pro Quelle laden
    buckets_by_source = load_buckets_by_source(bucket_stats_path, round_decimals=round_decimals)
    print(f"{len(buckets_by_source)} Quellen aus CSV geladen.")

    # predictions_ranking.json laden
    with pred_path.open("r", encoding="utf-8") as f:
        predictions = json.load(f)

    if not isinstance(predictions, list):
        raise ValueError("predictions_ranking.json muss eine Liste von Objekten enthalten.")

    # Pro Quelle: Indizes und y_pred sammeln
    src_indices: Dict[str, List[int]] = defaultdict(list)
    src_y_preds: Dict[str, List[float]] = defaultdict(list)

    for idx, entry in enumerate(predictions):
        src = entry.get("source") or entry.get("data_source")
        if src is None:
            raise ValueError(f"Eintrag {idx} hat weder 'source' noch 'data_source'.")
        if src not in buckets_by_source:
            raise ValueError(f"Quelle '{src}' nicht in bucket_stats_overall.csv gefunden.")

        yp = round(float(entry["y_pred"]), round_decimals)
        src_indices[src].append(idx)
        src_y_preds[src].append(yp)

    total_entries = len(predictions)
    mismatch_true = 0
    mismatch_pred = 0
    examples_true = []
    examples_pred = []

    # Jetzt pro Quelle erwartete Ranks berechnen und vergleichen
    for src, idx_list in src_indices.items():
        buckets = buckets_by_source[src]
        y_preds = src_y_preds[src]

        truth_rank_map, pred_rank_map = build_rank_maps_for_source(buckets, y_preds)

        for idx in idx_list:
            entry = predictions[idx]
            yt = round(float(entry["y_true"]), round_decimals)
            yp = round(float(entry["y_pred"]), round_decimals)

            stored_r_true = entry.get("ranking_true")
            stored_r_pred = entry.get("ranking_pred")

            # erwartete Ranks
            if yt not in truth_rank_map:
                raise ValueError(
                    f"y_true={yt} fuer source '{src}' nicht in Buckets-Liste. "
                    "CSV und Predictions passen evtl. nicht zusammen."
                )
            exp_r_true = truth_rank_map[yt]

            if yp not in pred_rank_map:
                raise ValueError(
                    f"y_pred={yp} fuer source '{src}' nicht in union(Buckets, y_pred). "
                    "Das sollte eigentlich nie passieren."
                )
            exp_r_pred = pred_rank_map[yp]

            if stored_r_true != exp_r_true:
                mismatch_true += 1
                if len(examples_true) < max_print:
                    examples_true.append(
                        {
                            "idx": idx,
                            "source": src,
                            "y_true": yt,
                            "stored_ranking_true": stored_r_true,
                            "expected_ranking_true": exp_r_true,
                        }
                    )

            if stored_r_pred != exp_r_pred:
                mismatch_pred += 1
                if len(examples_pred) < max_print:
                    examples_pred.append(
                        {
                            "idx": idx,
                            "source": src,
                            "y_pred": yp,
                            "stored_ranking_pred": stored_r_pred,
                            "expected_ranking_pred": exp_r_pred,
                        }
                    )

    # Zusammenfassung ausgeben
    print("\n=== Zusammenfassung ===")
    print(f"Total Eintraege:           {total_entries}")
    print(f"Abweichungen ranking_true: {mismatch_true}")
    print(f"Abweichungen ranking_pred: {mismatch_pred}")

    if mismatch_true > 0:
        print("\nBeispiele fuer Abweichungen bei ranking_true:")
        for ex in examples_true:
            print(
                f"  idx={ex['idx']}, source={ex['source']}, "
                f"y_true={ex['y_true']}, "
                f"stored={ex['stored_ranking_true']}, expected={ex['expected_ranking_true']}"
            )

    if mismatch_pred > 0:
        print("\nBeispiele fuer Abweichungen bei ranking_pred:")
        for ex in examples_pred:
            print(
                f"  idx={ex['idx']}, source={ex['source']}, "
                f"y_pred={ex['y_pred']}, "
                f"stored={ex['stored_ranking_pred']}, expected={ex['expected_ranking_pred']}"
            )


if __name__ == "__main__":
    main()
