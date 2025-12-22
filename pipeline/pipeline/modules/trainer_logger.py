from __future__ import annotations
from pathlib import Path
from typing import Dict, Any, Optional, List
import csv
import time

try:
    import mlflow
except Exception:
    mlflow = None


class TrainerLogger:
    """
    CSV-Logger für Training/Evaluation.

    Standardpfad:
        artifacts/logs/trainer_logs/<experiment_name>/trainer_logs.csv

    Steuerung über logger-Config (global_config + logger_config.json):
      - enabled: bool
      - backend: "csv" | "csv_mlflow" | "mlflow"
      - save_interval: int (für zukünftige Erweiterungen)
      - log_dir: optional, relativ zum Projektroot oder absolut
      - filename: optional, Name der CSV-Datei
    """

    def __init__(self, cfg: Dict[str, Any], global_cfg: Any):
        self.enabled: bool = bool(cfg.get("enabled", True))
        self.backend: str = cfg.get("backend", "csv")
        self.save_interval: int = int(cfg.get("save_interval", 1))

        # immer initialisieren, damit __del__ nie knallt
        self._file: Optional[Any] = None
        self._writer: Optional[csv.DictWriter] = None
        self._fieldnames: Optional[List[str]] = None

        # Basis-Pfade aus GlobalConfig
        artifacts_root = Path(global_cfg.paths.artifacts_root)  # z.B. .../pipeline/artifacts
        project_root = artifacts_root.parent                    # z.B. .../pipeline
        experiment_name = global_cfg.experiment_name

        # Standard-Logdir: artifacts/logs/trainer_logs/<experiment_name>
        default_logs_root = artifacts_root / "logs" / "trainer_logs"

        log_dir_cfg = cfg.get("log_dir")
        if log_dir_cfg:
            tmp = Path(log_dir_cfg)
            if tmp.is_absolute():
                base_log_dir = tmp
            else:
                # Relativ: interpretiere relativ zum Projektroot
                # z.B. "artifacts/logs/trainer_logs" -> <project_root>/artifacts/logs/trainer_logs
                base_log_dir = (project_root / tmp).resolve()
        else:
            base_log_dir = default_logs_root

        self.log_dir = base_log_dir / experiment_name
        self.log_dir.mkdir(parents=True, exist_ok=True)

        filename = cfg.get("filename", "trainer_logs.csv")
        self.log_path: Path = self.log_dir / filename

        print(f"[TrainerLogger] Logging to: {self.log_path}")

        self._mlflow_available = mlflow is not None and self.backend in ("mlflow", "csv_mlflow")

    # ---------------- interne Helfer ----------------

    def _open_new_file(
        self,
        fieldnames: List[str],
        existing_rows: Optional[List[Dict[str, Any]]] = None,
    ):
        """
        Öffnet die CSV neu, schreibt Header + optional bestehende Zeilen.
        """
        if self._file is not None:
            try:
                self._file.close()
            except Exception:
                pass

        self._file = open(self.log_path, "w", newline="", encoding="utf-8")
        self._writer = csv.DictWriter(self._file, fieldnames=fieldnames)
        self._writer.writeheader()

        if existing_rows:
            for row in existing_rows:
                full_row = {fn: row.get(fn, "") for fn in fieldnames}
                self._writer.writerow(full_row)

        self._fieldnames = fieldnames

    def _ensure_writer_with_row(self, row: Dict[str, Any]):
        """
        Stellt sicher, dass Writer und Header zu den Keys im Row-Dict passen.
        Falls neue Spalten auftauchen, wird die Datei komplett neu geschrieben
        (Header = Vereinigung aller bisher gesehenen Spalten).
        """
        keys = set(row.keys())

        # Fall 1: erste Zeile → neuen Header aufbauen
        if self._writer is None or self._fieldnames is None:
            fieldnames = sorted(keys)
            print(f"[TrainerLogger] Initializing CSV with columns: {fieldnames}")
            self._open_new_file(fieldnames)
            return

        # Fall 2: alle Keys sind bereits im Header → nichts zu tun
        current = set(self._fieldnames)
        if keys.issubset(current):
            return

        # Fall 3: neue Spalten → alte Datei einlesen, Header erweitern und neu schreiben
        new_fieldnames = sorted(current.union(keys))
        print(f"[TrainerLogger] Extending CSV columns to: {new_fieldnames}")

        existing_rows: List[Dict[str, Any]] = []
        try:
            if self.log_path.exists():
                with open(self.log_path, "r", encoding="utf-8") as f:
                    reader = csv.DictReader(f)
                    for r in reader:
                        existing_rows.append(r)
        except Exception as e:
            print(f"[TrainerLogger][WARN] Failed to read existing log file: {e}")
            existing_rows = []

        self._open_new_file(new_fieldnames, existing_rows=existing_rows)

    # ---------------- Public API ----------------

    def log_metrics(self, metrics: Dict[str, Any], step: int):
        """
        Wird vom BaseTrainer nach jeder Epoche (und für die Sanity-Zeile) aufgerufen:
          logger.log_metrics(metrics, step=epoch)
        """
        if not self.enabled:
            return

        # Basis-Row bauen
        row: Dict[str, Any] = {
            "step": step,
            "timestamp": time.time(),
        }

        for k, v in metrics.items():
            # step explizit überschreiben, falls vom Caller gesetzt
            if k == "step":
                try:
                    row["step"] = int(v)
                    continue
                except Exception:
                    pass
            try:
                row[k] = float(v)
            except (TypeError, ValueError):
                row[k] = str(v)

        # CSV
        if self.backend in ("csv", "csv_mlflow"):
            try:
                self._ensure_writer_with_row(row)
                assert self._writer is not None and self._fieldnames is not None

                full_row = {fn: row.get(fn, "") for fn in self._fieldnames}
                self._writer.writerow(full_row)
                if self._file is not None:
                    self._file.flush()

                # kleines Debug-Feedback
                if step == 0:
                    print(f"[TrainerLogger] Wrote sanity row to {self.log_path}")
                elif step == 1:
                    print(f"[TrainerLogger] Wrote first epoch row (step=1) to {self.log_path}")
            except Exception as e:
                print(f"[TrainerLogger][ERROR] Failed to write CSV row: {e}")
                raise

        # MLflow (optional)
        if self._mlflow_available:
            try:
                numeric_metrics = {
                    k: float(v)
                    for k, v in metrics.items()
                    if isinstance(v, (int, float))
                }
                if numeric_metrics:
                    mlflow.log_metrics(numeric_metrics, step=step)
            except Exception as e:
                print(f"[TrainerLogger][WARN] MLflow logging failed: {e}")

    def close(self):
        f = getattr(self, "_file", None)
        if f is not None:
            try:
                f.close()
            except Exception:
                pass

    def __del__(self):
        self.close()
