from __future__ import annotations

from pathlib import Path
import numpy as np
import torch

from pipeline.config import load_global_config
from pipeline.modules.dataloader import QADataModule


def compute_baseline_metrics(y_true: np.ndarray, y_pred: np.ndarray, name: str = "baseline"):
    """
    Berechne MSE, RMSE, MAE und R² für eine gegebene Vorhersage y_pred.
    Alles auf der normalisierten Skala [0, 1].
    """
    assert y_true.shape == y_pred.shape

    # MSE
    mse = float(((y_true - y_pred) ** 2).mean())
    rmse = float(np.sqrt(mse))
    mae = float(np.abs(y_true - y_pred).mean())

    # R²
    y_mean = float(y_true.mean())
    ss_res = float(((y_true - y_pred) ** 2).sum())
    ss_tot = float(((y_true - y_mean) ** 2).sum())
    r2 = float("nan") if ss_tot == 0.0 else 1.0 - ss_res / ss_tot

    print(f"\n📊 Ergebnisse für {name}:")
    print(f"  MSE  : {mse:.6f}")
    print(f"  RMSE : {rmse:.6f}")
    print(f"  MAE  : {mae:.6f}")
    print(f"  R²   : {r2:.6f}")

    return {
        "mse": mse,
        "rmse": rmse,
        "mae": mae,
        "r2": r2,
    }


def load_val_labels(global_cfg_path: Path, label_field: str = "normalized_grade") -> np.ndarray:
    """
    Lädt alle Labels aus dem Validation-Loader (genau so wie beim Training),
    und gibt sie als 1D-NumPy-Array zurück.
    """
    datamodule = QADataModule(global_config_path=global_cfg_path)
    val_loader = datamodule.val_loader()

    all_labels = []

    for batch in val_loader:
        # batch[label_field] kann Tensor oder Liste sein
        labels = batch[label_field]

        if isinstance(labels, torch.Tensor):
            labels_np = labels.detach().cpu().view(-1).numpy()
        else:
            labels_np = np.array(labels, dtype=np.float32).reshape(-1)

        all_labels.append(labels_np)

    if not all_labels:
        raise RuntimeError("Validation-Loader hat keine Batches geliefert – check deine Datenkonfiguration.")

    y_true = np.concatenate(all_labels, axis=0)
    print(f"✅ Geladene Val-Labels: {y_true.shape[0]} Samples, "
          f"mean={y_true.mean():.4f}, std={y_true.std():.4f}")
    return y_true


def main():
    # Pfad zur global_config.json – analog zu deinem main() im Training
    here = Path(__file__).resolve().parent
    global_cfg_path = here / "global_config.json"

    if not global_cfg_path.exists():
        raise FileNotFoundError(f"global_config.json nicht gefunden unter: {global_cfg_path}")

    # Config nur laden, damit QADataModule weiß, wo die Daten sind
    _ = load_global_config(global_cfg_path)

    # Alle Val-Labels laden
    y_true = load_val_labels(global_cfg_path, label_field="normalized_grade")

    # 1) Baseline: Immer den Mittelwert der Val-Labels vorhersagen
    mean_label = float(y_true.mean())
    y_pred_mean = np.full_like(y_true, mean_label)
    compute_baseline_metrics(y_true, y_pred_mean, name=f"Mean-Baseline (μ={mean_label:.4f})")

    # 2) Optional: Konstante Baseline 0.5
    y_pred_05 = np.full_like(y_true, 0.5)
    compute_baseline_metrics(y_true, y_pred_05, name="Konstante Baseline (0.5)")

    # 3) Optional: Konstante Baseline 0.75 (z.B. „meist gute Noten“)
    y_pred_075 = np.full_like(y_true, 0.75)
    compute_baseline_metrics(y_true, y_pred_075, name="Konstante Baseline (0.75)")


if __name__ == "__main__":
    np.random.seed(42)
    torch.manual_seed(42)
    main()
