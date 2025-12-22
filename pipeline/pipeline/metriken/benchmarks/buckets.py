#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from __future__ import annotations

from pathlib import Path
import pandas as pd
import json


def load_split(base_dir: Path, filename: str, split_name: str) -> pd.DataFrame:
    """Lade einen Parquet-Split und fuege eine Spalte 'split' hinzu."""
    path = base_dir / filename
    if not path.exists():
        raise FileNotFoundError(f"Datei nicht gefunden: {path}")
    df = pd.read_parquet(path)
    df["split"] = split_name
    return df


def main() -> None:
    # Dein absoluter Pfad
    base_dir = Path("/home/kevin/projektarbeit/pipeline/trainingsdata/ASAG2024/student_ranking")

    split_files = {
        "train": "rank_train.parquet",
        "validation": "rank_validation.parquet",
        "test": "rank_test.parquet",
    }

    # ---------------------------
    # 1) Daten laden und mergen
    # ---------------------------
    dfs = []
    for split_name, fname in split_files.items():
        df = load_split(base_dir, fname, split_name)
        dfs.append(df)

    df_all = pd.concat(dfs, ignore_index=True)

    # Check, ob relevante Spalten da sind
    cols_needed = ["data_source", "normalized_grade", "split"]
    missing = [c for c in cols_needed if c not in df_all.columns]
    if missing:
        raise KeyError(f"Fehlende Spalten in den Parquet-Dateien: {missing}")

    # ------------------------------------------------
    # 2) Bucket-Spalte aus normalized_grade ableiten
    #    -> nur Werte, die tatsaechlich vorkommen
    #    -> konsistent gerundet, um Floating-Noise zu vermeiden
    # ------------------------------------------------
    # Du kannst die Rundungstiefe bei Bedarf anpassen (z. B. 3 oder 4)
    df_all["bucket"] = df_all["normalized_grade"].astype(float).round(5)

    # -------------------------------------
    # 3) CSV-Statistiken (optional, aber
    #    oft praktisch fuer einen schnellen Blick)
    # -------------------------------------
    buckets_per_split = (
        df_all.groupby(["data_source", "split", "bucket"])
        .size()
        .reset_index(name="count")
        .sort_values(["data_source", "split", "bucket"])
    )
    buckets_per_split.to_csv(base_dir / "bucket_stats_per_split.csv", index=False)

    buckets_overall = (
        df_all.groupby(["data_source", "bucket"])
        .size()
        .reset_index(name="count")
        .sort_values(["data_source", "bucket"])
    )
    buckets_overall.to_csv(base_dir / "bucket_stats_overall.csv", index=False)

    # Zusammenfassung pro data_source
    g = df_all.groupby("data_source")["bucket"]
    summary = (
        pd.DataFrame(
            {
                "num_buckets": g.nunique(),
                "buckets": g.apply(
                    lambda s: sorted({float(v) for v in s.unique()})
                ),
            }
        )
        .reset_index()
        .sort_values("data_source")
    )
    summary.to_csv(base_dir / "bucket_summary_per_datasource.csv", index=False)

    # -------------------------------------
    # 4) JSON-Struktur aufbauen
    # -------------------------------------
    result = {}

    for ds in df_all["data_source"].unique():
        df_ds = df_all[df_all["data_source"] == ds].copy()

        # Liste der wirklich vorkommenden Buckets
        buckets = sorted({float(v) for v in df_ds["bucket"].unique()})
        num_buckets = len(buckets)

        # Verteilung pro Split
        per_split = {}
        for split_name in split_files.keys():
            df_split = df_ds[df_ds["split"] == split_name]
            counts = (
                df_split.groupby("bucket")
                .size()
                .to_dict()
            )
            # Keys als Strings fuer JSON
            per_split[split_name] = {str(float(k)): int(v) for k, v in counts.items()}

        # Gesamtverteilung ueber alle Splits
        overall_counts = (
            df_ds.groupby("bucket")
            .size()
            .to_dict()
        )
        overall_counts = {str(float(k)): int(v) for k, v in overall_counts.items()}

        result[ds] = {
            "num_buckets": num_buckets,
            "buckets": buckets,          # nur vorkommende Werte
            "per_split": per_split,
            "overall": overall_counts,
        }

    # -------------------------------------
    # 5) JSON schreiben
    # -------------------------------------
    json_path = base_dir / "bucket_summary.json"
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(result, f, indent=4)

    print(f"Fertig. JSON geschrieben nach: {json_path}")


if __name__ == "__main__":
    main()
