from __future__ import annotations

from pathlib import Path
import math
import numpy as np
import torch
from tqdm import tqdm

from pipeline.config import load_global_config
from pipeline.modules.dataloader import QADataModule
from pipeline.trainers.trainer_factory import TrainerFactory
from pipeline.modules.model_loader import build_model_loader

from transformers import AutoModelForSequenceClassification, AutoTokenizer, BitsAndBytesConfig


class ModelAdapter:
    """Light wrapper around model/tokenizer – identisch zu deinem main()."""
    def __init__(self, model, tokenizer):
        self.model = model
        self.tokenizer = tokenizer


def build_trainer(global_cfg_path: Path):
    """
    Baut Model + Datamodule + Trainer so wie in deinem main(), aber ohne train().
    """
    g = load_global_config(global_cfg_path)
    run_dirs = g.resolve_run_dirs("run")

    # --- Model laden (wie in main.py) ---
    model_loader = build_model_loader(g, run_dirs)
    model_artifact = model_loader.download()
    model_path = model_artifact.local_dir

    amp_dtype = (
        torch.bfloat16
        if (torch.cuda.is_available() and torch.cuda.is_bf16_supported())
        else torch.float16
    )

    bnb_cfg = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_use_double_quant=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=amp_dtype,
    )

    tokenizer = AutoTokenizer.from_pretrained(model_path, use_fast=False)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "right"

    model = AutoModelForSequenceClassification.from_pretrained(
        model_path,
        num_labels=1,
        problem_type="regression",
        quantization_config=bnb_cfg,
        device_map="auto",
        torch_dtype=amp_dtype,
        low_cpu_mem_usage=True,
    )

    if hasattr(model.config, "use_cache"):
        model.config.use_cache = False
    if hasattr(model.config, "pad_token_id") and tokenizer.pad_token_id is not None:
        model.config.pad_token_id = tokenizer.pad_token_id

    model_adapter = ModelAdapter(model, tokenizer)

    # Data
    dm = QADataModule(global_config_path=global_cfg_path)

    # Trainer-Config wie im main()
    trainer_cfg = (
        g.modules.trainer.model_dump()
        if hasattr(g.modules.trainer, "model_dump")
        else g.modules.trainer.dict()
    )

    # Kein Logger/Plotter nötig für reine Eval
    trainer = TrainerFactory.create(
        "llama",
        model_adapter,
        dm,
        trainer_cfg,
        logger=None,
    )

    return trainer, dm


def eval_checkpoint(trainer, dm, ckpt_path: Path, label_field: str = "normalized_grade"):
    """
    Lädt einen Checkpoint in trainer.model + trainer.reg_head und evaluiert
    auf dem Validation-Set mit CLAMP auf [0,1].
    """
    print(f"\n🔄 Evaluating checkpoint: {ckpt_path}")
    ckpt = torch.load(ckpt_path, map_location="cpu")

    model_state = ckpt.get("model", ckpt)
    if hasattr(trainer.model, "model") and hasattr(trainer.model.model, "load_state_dict"):
        trainer.model.model.load_state_dict(model_state, strict=False)
    elif hasattr(trainer.model, "load_state_dict"):
        trainer.model.load_state_dict(model_state, strict=False)

    if "reg_head" in ckpt and hasattr(trainer, "reg_head"):
        try:
            trainer.reg_head.load_state_dict(ckpt["reg_head"], strict=False)
        except Exception as e:
            print(f"[WARN] Could not load reg_head from checkpoint: {e}")

    trainer.model.model.eval()
    trainer.reg_head.eval()

    loader = dm.val_loader()

    all_preds, all_labels = [], []
    losses = []

    with torch.no_grad():
        pbar = tqdm(loader, desc=f"Validation ({ckpt_path.name})", dynamic_ncols=True, leave=False)
        for batch in pbar:
            loss, preds, labels = trainer.compute_loss_and_preds(batch)
            losses.append(loss.item())
            all_preds.append(preds.float().cpu())
            all_labels.append(labels.float().cpu())

    preds = torch.cat(all_preds, dim=0).squeeze(-1)
    labels = torch.cat(all_labels, dim=0).squeeze(-1)

    # 🔒 Hier clampen wir NUR für Metriken
    preds_clamped = preds.clamp(0.0, 1.0)

    se = (preds_clamped - labels) ** 2
    ae = (preds_clamped - labels).abs()

    mse = se.mean().item()
    rmse = math.sqrt(mse)
    mae = ae.mean().item()

    # R²
    y_true = labels.numpy()
    y_pred = preds_clamped.numpy()
    y_mean = y_true.mean()
    ss_res = ((y_true - y_pred) ** 2).sum()
    ss_tot = ((y_true - y_mean) ** 2).sum()
    r2 = float("nan") if ss_tot == 0.0 else float(1.0 - ss_res / ss_tot)

    val_loss = sum(losses) / max(1, len(losses))

    print(f"📊 {ckpt_path.name}:")
    print(f"  val_loss (original loss): {val_loss:.6f}")
    print(f"  RMSE (clamped)          : {rmse:.6f}")
    print(f"  MAE (clamped)           : {mae:.6f}")
    print(f"  R²  (clamped)           : {r2:.6f}")

    return {
        "ckpt": ckpt_path.name,
        "epoch": ckpt.get("epoch", None),
        "val_loss": val_loss,
        "rmse": rmse,
        "mae": mae,
        "r2": r2,
    }


def main():
    # eval_checkpoints_clamped.py liegt in: projektarbeit/pipeline/
    here = Path(__file__).resolve().parent          # .../projektarbeit/pipeline
    project_root = here.parent                      # .../projektarbeit

    global_cfg_path = here / "global_config.json"   # pipeline/global_config.json
    ckpt_dir = project_root / "checkpoints"         # projektarbeit/checkpoints

    print(f"[INFO] Using global_config: {global_cfg_path}")
    print(f"[INFO] Using checkpoint dir: {ckpt_dir}")

    if not global_cfg_path.exists():
        raise FileNotFoundError(f"global_config.json nicht gefunden unter: {global_cfg_path}")
    if not ckpt_dir.exists():
        raise FileNotFoundError(f"Checkpoint-Verzeichnis nicht gefunden: {ckpt_dir}")

    # Trainer & Data aufbauen
    trainer, dm = build_trainer(global_cfg_path)

    # Alle Checkpoints finden: model_epoch*.pt
    ckpt_paths = sorted(
        ckpt_dir.glob("model_epoch*.pt"),
        key=lambda p: int(p.stem.replace("model_epoch", "")),
    )
    if not ckpt_paths:
        raise RuntimeError(f"Keine Checkpoints in {ckpt_dir} gefunden.")

    results = []
    for ckpt_path in ckpt_paths:
        metrics = eval_checkpoint(trainer, dm, ckpt_path)
        results.append(metrics)

    # Am Schluss den besten Checkpoint nach clamped RMSE/R² ausgeben
    best_rmse = min(results, key=lambda m: m["rmse"])
    best_r2 = max(results, key=lambda m: m["r2"])

    print("\n🏆 Bestes Checkpoint nach RMSE (clamped):")
    print(
        f"  {best_rmse['ckpt']} | epoch={best_rmse['epoch']} "
        f"| RMSE={best_rmse['rmse']:.6f} | R²={best_rmse['r2']:.6f}"
    )

    print("\n🏆 Bestes Checkpoint nach R² (clamped):")
    print(
        f"  {best_r2['ckpt']} | epoch={best_r2['epoch']} "
        f"| RMSE={best_r2['rmse']:.6f} | R²={best_r2['r2']:.6f}"
    )


if __name__ == "__main__":
    torch.manual_seed(42)
    np.random.seed(42)
    main()
