from __future__ import annotations
from pathlib import Path
import os
import torch

from pipeline.config import (
    load_global_config,
    effective_logger_config,
    effective_plotter_config,
)
from pipeline.modules.model_loader import build_model_loader
from pipeline.modules.dataloader import QADataModule
from pipeline.trainers.trainer_factory import TrainerFactory
from pipeline.modules.trainer_logger import TrainerLogger
from pipeline.modules.plotter import TrainingPlotter


class ModelAdapter:
    """Light wrapper around model/tokenizer for saving/loading."""
    def __init__(self, model, tokenizer):
        self.model = model
        self.tokenizer = tokenizer

    def save(self, path: str | Path):
        path = Path(path)
        os.makedirs(path, exist_ok=True)
        self.model.save_pretrained(path)
        self.tokenizer.save_pretrained(path)


def main():
    print("🚀 Starting pipeline")

    # --- Load config and prepare run dirs ---
    here = Path(__file__).resolve().parent
    global_cfg_path = here / "global_config.json"

    g = load_global_config(global_cfg_path)
    run_dirs = g.resolve_run_dirs("run")  # bleibt so, wird weiter unten für Modelle benutzt

    # --- Logger & Plotter aus GlobalConfig bauen (WICHTIG: zweites Argument ist g, NICHT run_dirs) ---
    logger_cfg = effective_logger_config(g)
    trainer_logger = TrainerLogger(logger_cfg, g)

    plotter_cfg = effective_plotter_config(g)
    plotter = TrainingPlotter(plotter_cfg, g)

    # --- Step 1: Download / ensure model files ---
    model_loader = build_model_loader(g, run_dirs)
    model_artifact = model_loader.download()
    print(f"✅ Model available at: {model_artifact.local_dir}")

    # --- Step 2: Load and prepare dataset ---
    datamodule = QADataModule(global_config_path=global_cfg_path)
    train_loader = datamodule.train_loader()
    val_loader = datamodule.val_loader()
    print(f"✅ Data ready ({len(train_loader.dataset)} train / {len(val_loader.dataset)} val samples)")

    # --- Step 3: Initialize model as REGRESSION classifier (4-bit) ---
    from transformers import (
        AutoModelForSequenceClassification,
        AutoTokenizer,
        BitsAndBytesConfig,
    )

    model_path = model_artifact.local_dir

    # Choose compute dtype
    amp_dtype = torch.bfloat16 if (torch.cuda.is_available() and torch.cuda.is_bf16_supported()) else torch.float16

    # BitsAndBytes 4-bit config
    bnb_cfg = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_use_double_quant=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=amp_dtype,
    )

    # Tokenizer
    tokenizer = AutoTokenizer.from_pretrained(model_path, use_fast=False)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "right"

    # IMPORTANT: load as SequenceClassification with regression head
    model = AutoModelForSequenceClassification.from_pretrained(
        model_path,
        num_labels=1,
        problem_type="regression",
        quantization_config=bnb_cfg,
        device_map="auto",
        torch_dtype=amp_dtype,
        low_cpu_mem_usage=True,
    )

    # Memory-friendly tweaks
    if hasattr(model.config, "use_cache"):
        model.config.use_cache = False  # reduce VRAM during training

    # Ensure pad id is set on the model config
    if hasattr(model.config, "pad_token_id") and tokenizer.pad_token_id is not None:
        model.config.pad_token_id = tokenizer.pad_token_id

    # Slight speedup on Ampere/Lovelace
    try:
        torch.backends.cuda.matmul.allow_tf32 = True
    except Exception:
        pass

    model_adapter = ModelAdapter(model, tokenizer)

    # --- Step 4: Trainer selection ---
    trainer_cfg = (
        g.modules.trainer.model_dump()
        if hasattr(g.modules.trainer, "model_dump")
        else g.modules.trainer.dict()
    )

    # WICHTIG: logger an den Trainer übergeben
    trainer = TrainerFactory.create(
        "llama",
        model_adapter,
        datamodule,
        trainer_cfg,
        logger=trainer_logger,
    )

    # --- Step 5: Training ---
    trainer.train()

    # --- Step 6: Save fine-tuned model ---
    trained_dir = (
        Path(g.paths.artifacts_root)
        / g.paths.models_subdir
        / "trainedmodels"
        / g.experiment_name
    )
    model_adapter.save(trained_dir)
    print(f"💾 Saved fine-tuned model to: {trained_dir.resolve()}")

    print("🧪 Running final evaluation on test set...")
    # --- Step 7: Evaluate on test set using unified BaseTrainer API ---
    test_metrics = trainer.evaluate_split("test")
    print("📊 Test Results:")
    for k, v in test_metrics.items():
        print(f"  {k}: {v:.4f}")

    # --- Step 8: Plots erzeugen ---
    if trainer_logger.enabled:
        plotter.plot_all(trainer_logger.log_path)

    print("🏁 Training and testing finished successfully.")


if __name__ == "__main__":
    os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
    os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")

    torch.manual_seed(42)
    main()
