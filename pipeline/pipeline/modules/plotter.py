from __future__ import annotations
from pathlib import Path
from typing import Dict, Any, List
import csv

import matplotlib.pyplot as plt


class TrainingPlotter:
    """
    Plots landen standardmäßig in:
        artifacts/plots/<experiment_name>/

    Config-Override:
      - plotter_cfg["output_dir"] (optional)
        * absolut: direkt benutzt
        * relativ: relativ zum Projektroot
    """

    def __init__(self, cfg: Dict[str, Any], global_cfg: Any):
        self.enabled = bool(cfg.get("enabled", True))

        artifacts_root = Path(global_cfg.paths.artifacts_root)  # z.B. .../pipeline/artifacts
        project_root = artifacts_root.parent                    # z.B. .../pipeline
        experiment_name = global_cfg.experiment_name

        output_dir_cfg = cfg.get("output_dir")

        if output_dir_cfg:
            tmp = Path(output_dir_cfg)
            if tmp.is_absolute():
                base_out = tmp
            else:
                # Relativ: interpretiere relativ zum Projektroot
                # d.h. "artifacts/plots" -> <project_root>/artifacts/plots
                base_out = (project_root / tmp).resolve()
        else:
            # Standard: artifacts/plots
            base_out = artifacts_root / "plots"

        self.output_dir = base_out / experiment_name
        self.output_dir.mkdir(parents=True, exist_ok=True)

    # ---------------- interne Helfer ----------------

    def _load_metrics(self, log_path: Path) -> List[Dict[str, float]]:
        rows: List[Dict[str, float]] = []
        with open(log_path, "r", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            for row in reader:
                parsed: Dict[str, float] = {}
                for k, v in row.items():
                    if v is None or v == "":
                        continue
                    try:
                        parsed[k] = float(v)
                    except ValueError:
                        # Strings ignorieren wir
                        pass
                rows.append(parsed)
        return rows

    def _extract_series(self, rows: List[Dict[str, float]], key: str):
        xs, ys = [], []
        for r in rows:
            if key in r and "step" in r:
                xs.append(r["step"])
                ys.append(r[key])
        return xs, ys

    def _plot(self, rows: List[Dict[str, float]], key: str, title: str, filename: str):
        xs, ys = self._extract_series(rows, key)
        if not xs:
            return
        plt.figure()
        plt.plot(xs, ys, marker="o")
        plt.xlabel("epoch")
        plt.ylabel(key)
        plt.title(title)
        plt.grid(True)
        out_path = self.output_dir / filename
        plt.savefig(out_path, bbox_inches="tight")
        plt.close()

    def _plot_pair(self, rows: List[Dict[str, float]], k1: str, k2: str, title: str, filename: str):
        xs1, ys1 = self._extract_series(rows, k1)
        xs2, ys2 = self._extract_series(rows, k2)
        if not xs1 and not xs2:
            return
        plt.figure()
        if xs1:
            plt.plot(xs1, ys1, marker="o", label=k1)
        if xs2:
            plt.plot(xs2, ys2, marker="o", label=k2)
        plt.xlabel("epoch")
        plt.ylabel("value")
        plt.title(title)
        plt.grid(True)
        plt.legend()
        out_path = self.output_dir / filename
        plt.savefig(out_path, bbox_inches="tight")
        plt.close()

    # ---------------- Public API ----------------

    def plot_all(self, log_path: Path):
        if not self.enabled:
            print("[Plotter] Disabled via config, skipping plots.")
            return
        if not log_path.exists():
            print(f"[Plotter] No log file found at {log_path}, skipping.")
            return

        rows = self._load_metrics(log_path)
        if not rows:
            print("[Plotter] Log file is empty, skipping.")
            return

        # 1) Loss
        self._plot_pair(rows, "train_loss", "val_loss", "Loss", "loss_over_epochs.png")

        # 2) Val-Metriken
        self._plot(rows, "val_rmse", "Validation RMSE", "val_rmse.png")
        self._plot(rows, "val_mae", "Validation MAE", "val_mae.png")
        self._plot(rows, "val_r2", "Validation R2", "val_r2.png")

        # 3) LR
        self._plot_pair(rows, "train_lr_head", "train_lr_lora", "Learning Rate", "lr_over_epochs.png")

        # 4) Head-Stats
        self._plot(rows, "train_head_weight_norm", "Head Weight Norm", "head_weight_norm.png")
        self._plot(rows, "train_head_bias_mean", "Head Bias Mean", "head_bias_mean.png")

        # 5) Pred/Label-Stats
        self._plot_pair(rows, "val_pred_mean", "val_label_mean", "Val Mean", "val_means.png")
        self._plot_pair(rows, "val_pred_min", "val_label_min", "Val Min", "val_mins.png")
        self._plot_pair(rows, "val_pred_max", "val_label_max", "Val Max", "val_maxs.png")

        print(f"[Plotter] Plots written to {self.output_dir.resolve()}")
