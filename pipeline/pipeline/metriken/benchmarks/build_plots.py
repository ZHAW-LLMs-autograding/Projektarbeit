from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from plot_style import apply_default_style, apply_axes_font, apply_legend_font


# --------------------------------------------------------------------
# KONFIGURATION
# --------------------------------------------------------------------

# Trainingsplots global an/aus
ENABLE_TRAIN_PLOTS: bool = True

# Mapping: Modellordner-Name -> Trainingslog
# CSV: Epoch-Log mit val_loss
# JSON: Step-Log (siehe BERT-Beispiel)
TRAIN_LOGS: Dict[str, Path] = {
    "llama3.1_base_lora": Path("trainer_logs_base.csv"),
    "llama3.1_instruct_lora": Path("trainer_logs_instruct.csv"),
    "bert": Path("bert/train_metrics.json"),
}

# R2 aus predictions.json berechnen?
ENABLE_R2_FROM_PREDICTIONS: bool = True

# Metriken aus metrics.json (MSE/RMSE/wRMSE) anzeigen?
ENABLE_SUMMARY_METRICS_JSON: bool = True

# Basisordner (dieses Skript liegt in benchmarks/)
ROOT_DIR = Path(__file__).resolve().parent


# --------------------------------------------------------------------
# HELFER
# --------------------------------------------------------------------

def prettify_model_name(name: str) -> str:
    """Ordnername in einen huebscheren Plot-Titel verwandeln."""
    pretty = name
    for suffix in ["_run", "_base", "_instruct", "_lora"]:
        if pretty.endswith(suffix):
            pretty = pretty[: -len(suffix)]
    pretty = pretty.replace("_", " ")
    return pretty


def compute_r2(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Nur R2 berechnen."""
    assert y_true.shape == y_pred.shape
    ss_res = float(np.sum((y_true - y_pred) ** 2))
    ss_tot = float(np.sum((y_true - np.mean(y_true)) ** 2))
    if ss_tot <= 0.0:
        return float("nan")
    return 1.0 - ss_res / ss_tot


# --------------------------------------------------------------------
# TRAININGSPLOTS
# --------------------------------------------------------------------

def load_train_log_csv(csv_path: Path) -> Optional[pd.DataFrame]:
    """CSV-Log mit Epoch / val_loss laden (wie bei LLaMA-Logs)."""
    if not csv_path.exists():
        print(f"[Train] Logdatei nicht gefunden: {csv_path}")
        return None
    df = pd.read_csv(csv_path)
    df_eval = df[df["val_loss"].notna()].copy()
    if df_eval.empty:
        print(f"[Train] Keine Eval-Zeilen in {csv_path}")
        return None
    return df_eval


def load_train_log_json_steps(json_path: Path) -> Optional[pd.DataFrame]:
    """JSON-Log wie bei BERT laden.

    Erlaubte Formate:
    1) dict mit Key "train/loss" -> Liste aus {step, tag, value}
    2) Liste von Events mit "tag" == "train/loss"

    Wir mitteln alle Werte pro Step.
    """
    if not json_path.exists():
        print(f"[Train] JSON-Log nicht gefunden: {json_path}")
        return None

    try:
        with json_path.open("r", encoding="utf-8") as f:
            data = json.load(f)
    except Exception as e:
        print(f"[Train] Fehler beim Lesen {json_path}: {e}")
        return None

    events: List[dict] = []

    if isinstance(data, dict):
        if "train/loss" in data and isinstance(data["train/loss"], list):
            events = data["train/loss"]
        else:
            tmp: List[dict] = []
            for v in data.values():
                if isinstance(v, list):
                    tmp.extend(v)
            events = [e for e in tmp if isinstance(e, dict) and e.get("tag") == "train/loss"]
    elif isinstance(data, list):
        events = [e for e in data if isinstance(e, dict) and e.get("tag") == "train/loss"]
    else:
        print(f"[Train] Unerwartiges JSON-Format in {json_path}")
        return None

    rows = []
    for entry in events:
        try:
            step = int(entry["step"])
            value = float(entry["value"])
            rows.append({"step": step, "train_loss": value})
        except Exception as e:
            print(f"[Train] Fehlerhafter Eintrag in {json_path}: {e}")

    if not rows:
        print(f"[Train] Keine train/loss Eintraege in {json_path}")
        return None

    df = pd.DataFrame(rows)
    df = df.groupby("step", as_index=False)["train_loss"].mean()
    df = df.sort_values("step")
    return df


def plot_training_curve_epochs(model_name: str, df_eval: pd.DataFrame, out_path: Path) -> None:
    """Plot fuer CSV-Epoch-Logs (Train/Val loss pro Epoch)."""
    apply_default_style()
    fig, ax = plt.subplots(figsize=(7, 4))

    if "train_loss" in df_eval.columns:
        line_train, = ax.plot(df_eval["epoch"], df_eval["train_loss"], marker="o", label="Train loss")
    ax_val, = ax.plot(df_eval["epoch"], df_eval["val_loss"], marker="o", label="Val loss")

    ax.set_xlabel("Epoch")
    ax.set_ylabel("Loss (MSE)")
    title = f"Training and validation loss – {prettify_model_name(model_name)}"
    legend = ax.legend()
    ax.grid(True, which="both", linestyle="--", linewidth=0.6, alpha=0.7)

    apply_axes_font(ax, title=title)
    apply_legend_font(legend)

    fig.savefig(out_path, bbox_inches="tight")
    plt.close(fig)
    print(f"[Train] Epoch-Plot fuer {model_name} gespeichert unter: {out_path}")


def plot_training_curve_steps(model_name: str, df_steps: pd.DataFrame, out_path: Path) -> None:
    """Plot fuer JSON-Step-Logs (train/loss pro Step)."""
    apply_default_style()
    fig, ax = plt.subplots(figsize=(7, 4))

    ax.plot(df_steps["step"], df_steps["train_loss"], marker="o", linewidth=1)

    ax.set_xlabel("Step")
    ax.set_ylabel("Train loss")
    title = f"Training loss over steps – {prettify_model_name(model_name)}"
    ax.grid(True, which="both", linestyle="--", linewidth=0.6, alpha=0.7)

    apply_axes_font(ax, title=title)

    fig.savefig(out_path, bbox_inches="tight")
    plt.close(fig)
    print(f"[Train] Step-Plot fuer {model_name} gespeichert unter: {out_path}")


# --------------------------------------------------------------------
# TEST: metrics.json
# --------------------------------------------------------------------

def read_summary_metrics(metrics_path: Path) -> Optional[Dict[str, float]]:
    """Liest MSE/RMSE/wRMSE aus metrics.json, falls vorhanden."""
    if not metrics_path.exists():
        return None
    try:
        with metrics_path.open("r", encoding="utf-8") as f:
            data = json.load(f)
    except Exception as e:
        print(f"[Metrics] Fehler beim Lesen {metrics_path}: {e}")
        return None

    result: Dict[str, float] = {}
    for key in ["MSE", "RMSE", "wRMSE"]:
        if key in data:
            try:
                result[key] = float(data[key])
            except Exception:
                pass
    return result if result else None


# --------------------------------------------------------------------
# TEST: predictions.json laden
# --------------------------------------------------------------------

def load_predictions(pred_path: Path) -> Tuple[np.ndarray, np.ndarray]:
    """Laedt y_true / y_pred aus predictions.json."""
    if not pred_path.exists():
        print(f"[Pred] predictions.json nicht gefunden: {pred_path}")
        return np.array([]), np.array([])

    try:
        with pred_path.open("r", encoding="utf-8") as f:
            raw = json.load(f)
    except Exception as e:
        print(f"[Pred] Fehler beim Lesen {pred_path}: {e}")
        return np.array([]), np.array([])

    if isinstance(raw, dict) and "predictions" in raw:
        records = raw["predictions"]
    else:
        records = raw

    y_true_list: List[float] = []
    y_pred_list: List[float] = []

    if not isinstance(records, list):
        print(f"[Pred] Unerwartetes Format in {pred_path}")
        return np.array([]), np.array([])

    for item in records:
        try:
            y_true_list.append(float(item["y_true"]))
            y_pred_list.append(float(item["y_pred"]))
        except Exception as e:
            print(f"[Pred] Fehler bei Eintrag in {pred_path}: {e}")

    if not y_true_list:
        return np.array([]), np.array([])

    return np.array(y_true_list, dtype=float), np.array(y_pred_list, dtype=float)


# --------------------------------------------------------------------
# ERROR HISTOGRAM
# --------------------------------------------------------------------

def plot_error_hist(model_name: str, y_true: np.ndarray, y_pred: np.ndarray, out_path: Path):
    if y_true.size == 0:
        print(f"[Error] Keine Daten fuer Error Histogram bei {model_name}")
        return

    errors = y_pred - y_true

    apply_default_style()
    fig, ax = plt.subplots(figsize=(6, 4))

    ax.hist(errors, bins=30, alpha=0.7, edgecolor="black")
    ax.axvline(0, color="red", linestyle="--", linewidth=1)

    ax.set_xlabel("Prediction error (y_pred - y_true)")
    ax.set_ylabel("Count")
    title = f"Error Distribution – {prettify_model_name(model_name)}"
    ax.grid(True, linestyle="--", alpha=0.5)

    apply_axes_font(ax, title=title)

    fig.savefig(out_path, bbox_inches="tight")
    plt.close(fig)
    print(f"[Error] Histogram gespeichert: {out_path}")


# --------------------------------------------------------------------
# GROUPED PREDICTION BOXPLOT
# --------------------------------------------------------------------

def plot_grouped_boxplot(model_name: str, y_true: np.ndarray, y_pred: np.ndarray, out_path: Path):
    if y_true.size == 0:
        print(f"[Boxplot] Keine Daten fuer Boxplot bei {model_name}")
        return

    classes = sorted(set(y_true.tolist()))
    data: List[np.ndarray] = []
    labels: List[str] = []

    for c in classes:
        preds = y_pred[y_true == c]
        if preds.size == 0:
            continue
        data.append(preds)
        labels.append(str(c))

    if not data:
        print(f"[Boxplot] Keine Daten nach Klassen gruppierbar bei {model_name}")
        return

    apply_default_style()
    fig, ax = plt.subplots(figsize=(6, 4))

    bp = ax.boxplot(data, labels=labels, patch_artist=True)
    for box in bp["boxes"]:
        box.set(facecolor="#d0e1ff")

    ax.set_xlabel("True label")
    ax.set_ylabel("Predicted score")
    title = f"Predictions per True Label – {prettify_model_name(model_name)}"
    ax.grid(True, linestyle="--", alpha=0.5)

    apply_axes_font(ax, title=title)

    fig.savefig(out_path, bbox_inches="tight")
    plt.close(fig)
    print(f"[Boxplot] Grouped Prediction Boxplot gespeichert: {out_path}")


# --------------------------------------------------------------------
# MODEL-ORDNER FINDEN
# --------------------------------------------------------------------

def iterate_model_dirs(root: Path) -> List[Path]:
    dirs = []

    # direkt unter benchmarks/
    for child in sorted(root.iterdir()):
        if child.is_dir() and child.name not in {"__pycache__", "vectormodels"}:
            dirs.append(child)

    # char wb extra hinzufügen
    char_wb = root / "vectormodels" / "char_wb_3_5_min2"
    if char_wb.exists():
        dirs.append(char_wb)

    return dirs


# --------------------------------------------------------------------
# MAIN
# --------------------------------------------------------------------

def main():
    model_dirs = iterate_model_dirs(ROOT_DIR)
    print(f"Gefundene Modellordner: {[d.name for d in model_dirs]}")

    for model_dir in model_dirs:
        model_name = model_dir.name
        print(f"\n=== Modell: {model_name} ===")

        # 1) Trainingsplot
        if ENABLE_TRAIN_PLOTS and model_name in TRAIN_LOGS:
            log_path = ROOT_DIR / TRAIN_LOGS[model_name]
            if log_path.suffix == ".csv":
                df_eval = load_train_log_csv(log_path)
                if df_eval is not None:
                    out_path = model_dir / "training_curve_epochs.png"
                    plot_training_curve_epochs(model_name, df_eval, out_path)
            elif log_path.suffix == ".json":
                df_steps = load_train_log_json_steps(log_path)
                if df_steps is not None:
                    out_path = model_dir / "training_curve_steps.png"
                    plot_training_curve_steps(model_name, df_steps, out_path)
            else:
                print(f"[Train] Unbekanntes Log-Format fuer {model_name}: {log_path}")
        else:
            print(f"[Train] Kein Trainingslog fuer {model_name} konfiguriert oder deaktiviert.")

        # 2) metrics.json
        metrics_path = model_dir / "metrics_deduplicate.json"
        if ENABLE_SUMMARY_METRICS_JSON:
            metrics = read_summary_metrics(metrics_path)
            if metrics is not None:
                print("[Metrics] Zusammenfassung aus metrics.json:")
                for k in ["MSE", "RMSE", "wRMSE"]:
                    if k in metrics:
                        print(f"  {k}: {metrics[k]:.6f}")
            else:
                print("[Metrics] Keine gueltigen Metriken gefunden.")
        else:
            print("[Metrics] Lesen von metrics.json deaktiviert.")

        # 3) predictions.json -> R2 + Error-Hist + Grouped Boxplot
        pred_path = model_dir / "predictions_ranking_clamped.json"
        y_true, y_pred = load_predictions(pred_path)

        if ENABLE_R2_FROM_PREDICTIONS and y_true.size > 0:
            r2 = compute_r2(y_true, y_pred)
            print(f"[Pred] R2 aus predictions.json: {r2:.6f}")
        elif ENABLE_R2_FROM_PREDICTIONS:
            print("[Pred] Keine Daten fuer R2 berechnet.")

        plot_error_hist(model_name, y_true, y_pred, model_dir / "error_hist.png")
        plot_grouped_boxplot(model_name, y_true, y_pred, model_dir / "grouped_boxplot.png")


if __name__ == "__main__":
    main()
