from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, List, Optional

import pandas as pd
from plot_style import plot_metrics_table


# -----------------------------------------------------------
# KONFIG: HIER NUR DEINE MODELL-ORDNER EINTRAGEN
# Du kannst hier z.B. sowohl "bert" als auch "benchmarks/bert"
# eintragen, beides wird korrekt aufgeloest.
# -----------------------------------------------------------

MODEL_DIRS = [
    "benchmarks/bert",
    "benchmarks/llama_base",
    "benchmarks/llama_instruct",
    "benchmarks/llama3.1_base_lora",
    "benchmarks/llama3.1_instruct_lora",
    "benchmarks/sentence_transformer",
    "benchmarks/vectormodels/char_wb_3_5_min2",
]

# Welche Werte sollen gelesen werden?
METRICS = ["MAE", "RMSE", "wRMSE", "TAU"]   # Erweiterbar

OUT_TABLE_CSV = Path("metrics_table.csv")
OUT_FIG_PATH = Path("metrics_table.png")
TABLE_TITLE = "Model Metric Comparison"

SCRIPT_DIR = Path(__file__).resolve().parent


# -----------------------------------------------------------
# HELFER
# -----------------------------------------------------------

def resolve_model_dir(folder: str) -> Optional[Path]:
    """Versucht, einen Modellordner aus verschiedenen Basen aufzueloesen.

    Erlaubt:
      - absolute Pfade
      - Pfade relativ zum Skriptordner (benchmarks/...)
      - Pfade relativ zum Parent des Skriptordners (../benchmarks/...)
    """
    p = Path(folder)

    candidates = []
    if p.is_absolute():
        candidates.append(p)
    else:
        # typisch: "bert" oder "llama_base_run"
        candidates.append(SCRIPT_DIR / p)
        # falls jemand "benchmarks/bert" eintraegt, obwohl das Skript in benchmarks liegt
        candidates.append(SCRIPT_DIR.parent / p)

    for c in candidates:
        if c.exists():
            return c

    print(f"Warnung: Ordner {folder!r} wurde nicht gefunden.")
    return None


def load_metrics(path: Path) -> Dict[str, float]:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def build_dataframe() -> pd.DataFrame:
    rows: List[Dict[str, float]] = []
    index: List[str] = []

    for folder in MODEL_DIRS:
        model_dir = resolve_model_dir(folder)
        if model_dir is None:
            continue

        metrics_file = model_dir / "metrics.json"
        if not metrics_file.exists():
            print(f"Warnung: Kein metrics.json gefunden in {model_dir}")
            continue

        raw = load_metrics(metrics_file)
        row = {metric: raw.get(metric, None) for metric in METRICS}

        rows.append(row)
        # fuer die Anzeige: letzte Ordnerkomponente; wenn du lieber den vollen
        # Eintrag aus MODEL_DIRS willst, nimm einfach `folder` statt `model_dir.name`
        index.append(model_dir.name)

    return pd.DataFrame(rows, index=index)


# -----------------------------------------------------------
# MAIN
# -----------------------------------------------------------

def main():
    df = build_dataframe()

    if df.empty:
        print("Es wurden keine gueltigen metrics.json Dateien gefunden. Breche ab.")
        return

    df.to_csv(OUT_TABLE_CSV, float_format="%.6f")
    print("Saved CSV:", OUT_TABLE_CSV)
    print()
    print("Preview:")
    print(df)

    fig, _ = plot_metrics_table(df, title=TABLE_TITLE)
    fig.savefig(OUT_FIG_PATH, bbox_inches="tight")
    print("Saved Figure:", OUT_FIG_PATH)


if __name__ == "__main__":
    main()
