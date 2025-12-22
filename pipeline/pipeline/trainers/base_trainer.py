from __future__ import annotations
from abc import ABC, abstractmethod
from typing import Any, Dict, Optional, Union, List
import os
import torch

MetricDict = Dict[str, float]
EvalReturn = Union[float, MetricDict]
TrainReturn = Union[float, MetricDict]


class Callback(ABC):
    def on_train_start(self, trainer: "BaseTrainer"): ...
    def on_train_end(self, trainer: "BaseTrainer"): ...
    def on_epoch_start(self, trainer: "BaseTrainer", epoch: int): ...
    def on_epoch_end(self, trainer: "BaseTrainer", epoch: int, metrics: MetricDict): ...
    def on_validation_end(self, trainer: "BaseTrainer", epoch: int, metrics: MetricDict): ...


class EarlyStopping(Callback):
    def __init__(self, monitor="loss", mode="min", patience=3, min_delta=0.0):
        self.monitor = monitor
        self.mode = mode
        self.patience = patience
        self.min_delta = min_delta
        self._best = None
        self._bad = 0
        self.should_stop = False

    def on_validation_end(self, trainer, epoch, metrics):
        value = metrics.get(self.monitor, None)
        if value is None:
            return
        better = (
            (lambda c, b: c < b - self.min_delta)
            if self.mode == "min"
            else (lambda c, b: c > b + self.min_delta)
        )
        if self._best is None or better(value, self._best):
            self._best = value
            self._bad = 0
        else:
            self._bad += 1
            if self._bad >= self.patience:
                self.should_stop = True


def _coalesce(val, default):
    """Gibt default zurück, wenn val None ist – sonst val."""
    return default if val is None else val


class BaseTrainer(ABC):
    """
    Modellunabhängiger Trainer:
    - orchestriert Epochen, Logging, Checkpoints, Callbacks
    - einheitliche Evaluation über alle Splits (train/val/test)
    """

    def __init__(
        self,
        cfg: Dict[str, Any],
        model_adapter,
        datamodule,
        logger: Optional[Any] = None,
        callbacks: Optional[List[Callback]] = None,
    ):
        self.cfg = cfg
        self.model = model_adapter
        self.dm = datamodule
        self.logger = logger
        self.callbacks: List[Callback] = callbacks or []

        # EarlyStopping aus Config automatisch hinzufügen (falls konfiguriert)
        es_cfg = self.cfg.get("early_stopping")
        if es_cfg is not None:
            self.callbacks.append(
                EarlyStopping(
                    monitor=es_cfg.get("monitor", "loss"),
                    mode=es_cfg.get("mode", "min"),
                    patience=es_cfg.get("patience", 3),
                    min_delta=es_cfg.get("min_delta", 0.0),
                )
            )

        dev = _coalesce(cfg.get("device"), "cuda" if torch.cuda.is_available() else "cpu")
        self.device = torch.device(dev)

        # Sanity-Konfiguration (optional)
        self.sanity_before_train = bool(_coalesce(self.cfg.get("sanity_before_train"), True))
        self.sanity_batches = int(_coalesce(self.cfg.get("sanity_batches"), 4))
        self.sanity_split = _coalesce(self.cfg.get("sanity_split"), "validation")  # "validation" | "train" | "test"

        # --- einheitliche Metrik-Optionen für evaluate_split ---
        # None = nutze Subklassen-Flag report_clamp_eval, sonst bool(metric_clamp01)
        self.metric_clamp01: Optional[bool] = self.cfg.get("metric_clamp01")
        self.metric_weight_field: Optional[str] = _coalesce(self.cfg.get("metric_weight_field"), None)
        self.metric_normalize_weights: bool = bool(
            _coalesce(self.cfg.get("metric_normalize_weights"), True)
        )

        # aktuell laufende Epoche (für Subklassen/Batches nutzbar)
        self.current_epoch: Optional[int] = None

    # ---- Schnittstellen, die Subklassen frei implementieren ----
    @abstractmethod
    def train_epoch(self) -> TrainReturn:
        """Trainiere genau EINE Epoche. Rückgabe: float (train_loss) ODER dict mit Metriken."""
        ...

    def predict_batch(self, batch) -> tuple[torch.Tensor, torch.Tensor]:
        raise NotImplementedError(
            "Subklasse muss entweder compute_loss_and_preds ODER predict_batch implementieren."
        )

    # ---- Einheitliche Evaluation für alle Trainer (inkl. Test!) ----
    @torch.no_grad()
    def evaluate_split(self, split: str = "validation") -> Dict[str, float]:
        if split in ("val", "validation"):
            loader = self.dm.val_loader()
        elif split == "train":
            loader = self.dm.train_loader()
        else:
            loader = self.dm.test_loader()

        losses: List[float] = []
        preds_list: List[torch.Tensor] = []
        labels_list: List[torch.Tensor] = []
        weights_list: List[torch.Tensor] = []

        # Clamp-Regel bestimmen
        if self.metric_clamp01 is None:
            clamp01 = bool(getattr(self, "report_clamp_eval", False))
        else:
            clamp01 = bool(self.metric_clamp01)

        weight_field = self.metric_weight_field

        for batch in loader:
            if hasattr(self, "compute_loss_and_preds"):
                loss, preds, labels = self.compute_loss_and_preds(batch)  # von Subklasse
                if isinstance(loss, torch.Tensor):
                    loss = float(loss.detach().cpu())
                losses.append(loss)
            else:
                preds, labels = self.predict_batch(batch)

            # Form & Device vereinheitlichen
            if isinstance(preds, torch.Tensor):
                preds = preds.detach().float().cpu().squeeze(-1)
            else:
                preds = torch.as_tensor(preds, dtype=torch.float32)

            if isinstance(labels, torch.Tensor):
                labels = labels.detach().float().cpu().squeeze(-1)
            else:
                labels = torch.as_tensor(labels, dtype=torch.float32)

            # Optional: Gewichte extrahieren (nur für Metriken)
            if (
                weight_field
                and isinstance(batch, dict)
                and (weight_field in batch)
                and batch[weight_field] is not None
            ):
                w = batch[weight_field]
                if not isinstance(w, torch.Tensor):
                    w = torch.as_tensor(w)
                w = w.detach().float().cpu().view(-1)
                # trim auf n (gleiche Länge wie preds/labels)
                n = min(preds.shape[0], labels.shape[0], w.shape[0])
                weights_list.append(w[:n])
                preds_list.append(preds[:n])
                labels_list.append(labels[:n])
            else:
                # Batch-Größen angleichen
                n = min(preds.shape[0], labels.shape[0])
                preds_list.append(preds[:n])
                labels_list.append(labels[:n])

        if not preds_list:
            return {"loss": float("nan"), "rmse": float("nan"), "mae": float("nan"), "r2": float("nan")}

        preds_all = torch.cat(preds_list, dim=0)
        labels_all = torch.cat(labels_list, dim=0)

        preds_for_metrics = preds_all
        # WICHTIG: clamp ist komplett raus – keine Verzerrung mehr!
        # if clamp01:
        #     preds_for_metrics = preds_for_metrics.clamp(0.0, 1.0)

        # --- Metriken (optional gewichtet) ---
        def _weighted_mean(x: torch.Tensor, w: Optional[torch.Tensor]) -> float:
            if w is None:
                return float(x.mean().item())
            num = (x * w).sum()
            den = w.sum().clamp_min(1e-12)
            return float((num / den).item())

        weights_all: Optional[torch.Tensor] = None
        if len(weights_list) > 0:
            weights_all = torch.cat(weights_list, dim=0)
            if self.metric_normalize_weights:
                # durch Mittelwert normalisieren → effektive Durchschnitts-Gewichtung ~1.0
                weights_all = weights_all / weights_all.mean().clamp_min(1e-12)

        se = (preds_for_metrics - labels_all) ** 2
        ae = (preds_for_metrics - labels_all).abs()

        if weights_all is None:
            mse = float(se.mean().item())
            mae = float(ae.mean().item())
        else:
            mse = _weighted_mean(se, weights_all)
            mae = _weighted_mean(ae, weights_all)

        rmse = mse ** 0.5
        r2 = self._r2_score(labels_all, preds_for_metrics)

        # loss_mean: wenn Subklassen-Loss bereitgestellt → Mittel der Batch-Losses,
        # sonst fallback auf (un)gewichtetes MSE
        loss_mean = float(sum(losses) / len(losses)) if losses else mse

        return {"loss": loss_mean, "rmse": rmse, "mae": mae, "r2": r2}

    def evaluate(self) -> EvalReturn:
        return self.evaluate_split("validation")

    @torch.no_grad()
    def _quick_sanity_eval(self):
        if not self.sanity_before_train:
            return
        try:
            if self.sanity_split in ("val", "validation"):
                loader = self.dm.val_loader()
            elif self.sanity_split == "train":
                loader = self.dm.train_loader()
            else:
                loader = self.dm.test_loader()
        except Exception:
            return

        print(f"🔎 Sanity check on '{self.sanity_split}' ({self.sanity_batches} batches):")
        shown = 0
        losses = []
        for batch in loader:
            if hasattr(self, "compute_loss_and_preds"):
                loss, preds, labels = self.compute_loss_and_preds(batch)
                losses.append(
                    float(loss.detach().cpu()) if isinstance(loss, torch.Tensor) else float(loss)
                )
                # kompakt anzeigen
                p = preds.detach().float().cpu().view(-1).tolist()
                y = labels.detach().float().cpu().view(-1).tolist()
                k = min(5, len(p))
                print(
                    f"  • batch_loss={losses[-1]:.6f} | preds[:{k}]={p[:k]} | labels[:{k}]={y[:k]}"
                )
            shown += 1
            if shown >= self.sanity_batches:
                break
        if losses:
            print(f"  ⇒ mean_loss={sum(losses)/len(losses):.6f}")

    def save_checkpoint(self, epoch: int):
        out_dir = _coalesce(self.cfg.get("checkpoint_dir"), "checkpoints")
        os.makedirs(out_dir, exist_ok=True)
        path = os.path.join(out_dir, f"model_epoch{epoch}.pt")

        # Speichere reine Tensoren (State Dicts) für Modell (+ optional Regressionskopf)
        state: Dict[str, Any] = {"epoch": epoch}

        if hasattr(self.model, "model") and hasattr(self.model.model, "state_dict"):
            state["model"] = self.model.model.state_dict()
        elif hasattr(self.model, "state_dict"):
            state["model"] = self.model.state_dict()
        else:
            # Fallback: gesamtes Objekt speichern (nicht ideal, aber besser als gar nichts)
            state["model"] = self.model

        # Optional: separater Kopf wie in LlamaTrainer (reg_head)
        if hasattr(self, "reg_head") and hasattr(self.reg_head, "state_dict"):
            state["reg_head"] = self.reg_head.state_dict()

        torch.save(state, path)
        print(f"✅ Saved checkpoint to {path}")

    # --- Helper: automatisch das neueste Epoch-Checkpoint finden ---
    def _get_latest_checkpoint_path(self) -> Optional[str]:
        out_dir = _coalesce(self.cfg.get("checkpoint_dir"), "checkpoints")
        if not os.path.isdir(out_dir):
            return None

        candidates = []
        for fname in os.listdir(out_dir):
            if not fname.startswith("model_epoch") or not fname.endswith(".pt"):
                continue
            core = fname[len("model_epoch") : -3]  # ".pt" weg
            try:
                ep = int(core)
                candidates.append((ep, os.path.join(out_dir, fname)))
            except ValueError:
                continue

        if not candidates:
            return None

        candidates.sort(key=lambda x: x[0])
        print("Loaded last checkpoint and starting from there: " + candidates[-1][1])
        return candidates[-1][1]  # Pfad mit größter Epoche

    # ---- Orchestrierung ----
    def train(self):
        epochs = int(_coalesce(self.cfg.get("epochs"), 3))
        save_interval = int(_coalesce(self.cfg.get("save_interval"), 1))

        start_epoch = int(_coalesce(self.cfg.get("start_epoch"), 1))
        resume_ckpt = self.cfg.get("resume_from_checkpoint")
        auto_resume = bool(_coalesce(self.cfg.get("auto_resume"), False))

        # Auto-Resume: wenn kein explizites resume_from_checkpoint, aber auto_resume=True
        if resume_ckpt is None and auto_resume:
            resume_ckpt = self._get_latest_checkpoint_path()
            if resume_ckpt is not None:
                print(f"🔄 auto_resume: using latest checkpoint '{resume_ckpt}'")

        if resume_ckpt is not None and os.path.exists(resume_ckpt):
            try:
                ckpt = torch.load(resume_ckpt, map_location="cpu") # geändert <----
                model_state = ckpt.get("model", ckpt)

                if hasattr(self.model, "model") and hasattr(self.model.model, "load_state_dict"):
                    self.model.model.load_state_dict(model_state, strict=False)
                elif hasattr(self.model, "load_state_dict"):
                    self.model.load_state_dict(model_state, strict=False)

                if "reg_head" in ckpt and hasattr(self, "reg_head"):
                    try:
                        self.reg_head.load_state_dict(ckpt["reg_head"], strict=False)
                    except Exception:
                        pass

                loaded_epoch = ckpt.get("epoch")
                if isinstance(loaded_epoch, int):
                    start_epoch = max(start_epoch, loaded_epoch + 1)

                print(f"🔄 Resumed from checkpoint '{resume_ckpt}' (starting at epoch {start_epoch})")
            except Exception as e:
                print(f"[WARN] Failed to resume from checkpoint '{resume_ckpt}': {e}")

        for cb in self.callbacks:
            cb.on_train_start(self)

        # Sanity vor dem eigentlichen Training
        self._quick_sanity_eval()

        # --- Logging-Sanity: direkt zu Beginn eine Dummy-Zeile schreiben ---
        if self.logger is not None:
            try:
                sanity_metrics: Dict[str, Any] = {
                    "epoch": 0,
                    "event": "train_start",
                }
                # optional: Device-Info mitschreiben
                try:
                    sanity_metrics["device_type"] = getattr(self.device, "type", str(self.device))
                    idx = getattr(self.device, "index", None)
                    if idx is not None:
                        sanity_metrics["device_index"] = int(idx)
                except Exception:
                    pass

                self.logger.log_metrics(sanity_metrics, step=0)
                print(
                    f"📝 Logger sanity: wrote initial row to "
                    f"{getattr(self.logger, 'log_path', '<unknown>')}"
                )
            except Exception as e:
                print(f"[WARN] Logger sanity failed: {e}")

        for epoch in range(start_epoch, epochs + 1):
            # aktuell laufende Epoche setzen (für Subklassen/Batches)
            self.current_epoch = epoch

            for cb in self.callbacks:
                cb.on_epoch_start(self, epoch)

            tr = self.train_epoch()
            if isinstance(tr, dict):
                train_metrics = {"train_" + k: float(v) for k, v in tr.items()}
            else:
                train_metrics = {"train_loss": float(tr)}

            ev = self.evaluate()
            if isinstance(ev, dict):
                val_metrics = {
                    "val_" + k if not k.startswith("val_") else k: float(v)
                    for k, v in ev.items()
                }
            else:
                val_metrics = {"val_loss": float(ev)}

            metrics = {"epoch": epoch, **train_metrics, **val_metrics}

            if self.logger:
                try:
                    self.logger.log_metrics(metrics, step=epoch)
                except Exception as e:
                    print(f"[WARN] logger.log_metrics failed at epoch {epoch}: {e}")

            msg = f"[Epoch {epoch}] " + " | ".join(
                [f"{k}={v:.4f}" for k, v in metrics.items() if k != "epoch"]
            )
            print(msg)

            # Für EarlyStopping: val_* → loss, rmse, mae, r2
            for cb in self.callbacks:
                cb.on_validation_end(
                    self,
                    epoch,
                    {k.replace("val_", ""): v for k, v in val_metrics.items()},
                )
            for cb in self.callbacks:
                cb.on_epoch_end(self, epoch, metrics)

            # Checkpointing, z. B. alle 2 Epochen
            if (epoch % save_interval) == 0:
                self.save_checkpoint(epoch)

            if any(getattr(cb, "should_stop", False) for cb in self.callbacks):
                print("⏹️ Early stopping triggered.")
                break

        for cb in self.callbacks:
            cb.on_train_end(self)

    @staticmethod
    def _r2_score(y_true: torch.Tensor, y_pred: torch.Tensor) -> float:
        y_mean = torch.mean(y_true)
        ss_res = torch.sum((y_true - y_pred) ** 2)
        ss_tot = torch.sum((y_true - y_mean) ** 2)
        if float(ss_tot.item()) == 0.0:
            return float("nan")
        return float(1.0 - (ss_res / ss_tot).item())
