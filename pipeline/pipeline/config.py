from __future__ import annotations
from pathlib import Path
from typing import Optional, List, Dict, Any, Literal
from pydantic import BaseModel, Field, validator
from copy import deepcopy
import json
import logging
import sys

logger = logging.getLogger("pipeline.config")
_handler = logging.StreamHandler(sys.stdout)
_handler.setFormatter(logging.Formatter("[%(levelname)s] %(message)s"))
logger.addHandler(_handler)
logger.setLevel(logging.INFO)

# ============================================================
# Base Config Structures
# ============================================================

class PathsCfg(BaseModel):
    artifacts_root: Path = Path("artifacts")
    data_root: Path = Path("data")
    cache_root: Path = Path(".cache")
    logs_root: Path = Path("logs")
    models_subdir: str = "models"                 # RELATIV zu artifacts_root
    configs_root: Path = Path("pipeline/configs") # Modul-Default-JSONs

class EnvCfg(BaseModel):
    device: Literal["cpu", "cuda", "mps"] = "cpu"
    num_workers: int = 2
    precision: Literal["fp32", "fp16", "bf16", "auto"] = "auto"
    deterministic: bool = False

class DatasetCfg(BaseModel):
    name: str
    splits: Dict[str, Any] = Field(default_factory=lambda: {"source": "dataset_fixed"})
    language: Optional[str] = None

class TrackingCfg(BaseModel):
    backend: Literal["mlflow", "wandb", "none"] = "none"
    project: Optional[str] = None
    tags: List[str] = Field(default_factory=list)

class SecretsCfg(BaseModel):
    hf_token_env: Optional[str] = "HF_TOKEN"

# ============================================================
# Module Config Schemas
# ============================================================

class ModelLoaderOverrides(BaseModel):
    model_id: Optional[str] = None
    revision: Optional[str] = None
    allow_patterns: Optional[List[str]] = None
    ignore_patterns: Optional[List[str]] = None
    trust_remote_code: Optional[bool] = None
    skip_if_present: Optional[bool] = None
    required_files: Optional[List[str]] = None
    force_download: Optional[bool] = None
    smoke_test: Optional[bool] = None

    class Config:
        extra = "forbid"

class DataSchemaCfg(BaseModel):
    id_field: Optional[str] = "__index_level_0__"
    question_field: str = "question"
    context_field: Optional[str] = None
    answer_field: str = "reference_answer"
    provided_answer_field: Optional[str] = "provided_answer"
    choices_field: Optional[str] = None
    weight_field: Optional[str] = "weight"
    passthrough_fields: List[str] = Field(default_factory=list)

class DataAugmentCfg(BaseModel):
    p_text_noise: float = 0.0
    p_negate_answer: float = 0.0
    p_shuffle_choices: float = 0.0
    noise_drop_prob: float = 0.0
    noise_swap_prob: float = 0.0
    noise_insert_prob: float = 0.0
    noise_on_question: bool = True
    noise_on_context: bool = False
    apply_on_train: bool = True
    apply_on_val: bool = False
    apply_on_test: bool = False
    enabled: bool = True

class DataLoaderCfg(BaseModel):
    batch_size: int = 32
    num_workers: int = 0
    shuffle_train: bool = False
    pin_memory: bool = True
    drop_last: bool = False

class DataLoaderOverrides(BaseModel):
    schema: Optional[DataSchemaCfg] = None
    augmentation: Optional[DataAugmentCfg] = None
    loader: Optional[DataLoaderCfg] = None

    class Config:
        extra = "forbid"

class TrainerCfg(BaseModel):
    # --- generisch ---
    type: str = "bert"
    epochs: int = 3
    lr: float = 5e-5
    optimizer: Literal["adamw", "adam", "sgd"] = "adamw"
    gradient_accumulation_steps: int = 1
    warmup_steps: int = 0
    weight_decay: float = 0.01
    max_grad_norm: float = 1.0
    mixed_precision: bool = False
    log_interval: int = 100
    save_interval: int = 1
    device: Literal["auto", "cpu", "cuda"] = "auto"
    fine_tune_strategy: Literal["full", "lora", "adapter", "prefix"] = "lora"

    # --- optionale Felder für Regressions-/LLM-Trainer (werden von anderen ignoriert) ---
    label_field: str = "normalized_grade"
    weight_field: Optional[str] = "weight"
    use_smooth_l1: bool = True
    smooth_l1_beta: float = 0.5
    max_length: int = 512

    # --- Llama3.1-Regression spezifisch ---
    output_min: float = 1.0
    output_max: float = 9.0
    labels_are_scaled: bool = True   # True: Labels schon in [0,1]
    debug_interval: int = 0          # 0 = keine Batch-Debug-Ausgabe

    # --- Checkpointing / Resume / Early Stopping (für BaseTrainer) ---
    checkpoint_dir: Optional[str] = None
    resume_from_checkpoint: Optional[str] = None
    start_epoch: int = 1
    early_stopping: Optional[Dict[str, Any]] = None
    auto_resume: bool = False

class TrainerOverrides(BaseModel):
    # --- generisch ---
    type: Optional[str] = None
    epochs: Optional[int] = None
    lr: Optional[float] = None
    optimizer: Optional[str] = None
    gradient_accumulation_steps: Optional[int] = None
    warmup_steps: Optional[int] = None
    weight_decay: Optional[float] = None
    max_grad_norm: Optional[float] = None
    mixed_precision: Optional[bool] = None
    log_interval: Optional[int] = None
    save_interval: Optional[int] = None
    device: Optional[str] = None
    fine_tune_strategy: Optional[str] = None

    # --- spezifisch/optional für Llama-Regression ---
    label_field: Optional[str] = None
    weight_field: Optional[str] = None
    use_smooth_l1: Optional[bool] = None
    smooth_l1_beta: Optional[float] = None
    max_length: Optional[int] = None

    # --- Skalen/Debug ---
    output_min: Optional[float] = None
    output_max: Optional[float] = None
    labels_are_scaled: Optional[bool] = None
    debug_interval: Optional[int] = None

    # --- Checkpointing / Resume / Early Stopping (für BaseTrainer) ---
    checkpoint_dir: Optional[str] = None
    resume_from_checkpoint: Optional[str] = None
    start_epoch: Optional[int] = None
    early_stopping: Optional[Dict[str, Any]] = None
    auto_resume: Optional[bool] = None

    class Config:
        extra = "forbid"

class LoggerOverrides(BaseModel):
    # wichtig: diese Felder müssen zu dem passen, was du in global_config.json angibst
    enabled: Optional[bool] = None
    backend: Optional[str] = None
    save_interval: Optional[int] = None
    log_dir: Optional[str] = None       # <- NEU: erlaubt "log_dir": ...
    filename: Optional[str] = None      # <- NEU: erlaubt "filename": ...

    class Config:
        extra = "forbid"

class PlotterOverrides(BaseModel):
    enabled: Optional[bool] = None
    output_dir: Optional[str] = None

    class Config:
        extra = "forbid"

# ============================================================
# Modules registry in GlobalConfig
# ============================================================

class ModulesCfg(BaseModel):
    model_loader: Optional[ModelLoaderOverrides] = None
    data_loader: Optional[DataLoaderOverrides] = None
    trainer: Optional[TrainerOverrides] = None
    logger: Optional[LoggerOverrides] = None
    plotter: Optional[PlotterOverrides] = None

# ============================================================
# Global Config
# ============================================================

class GlobalConfig(BaseModel):
    experiment_name: str
    run_id_mode: Literal["timestamp", "git", "timestamp_git"] = "timestamp"
    seed: int = 42

    paths: PathsCfg = PathsCfg()
    environment: EnvCfg = EnvCfg()
    dataset: DatasetCfg
    tracking: TrackingCfg = TrackingCfg()
    secrets: SecretsCfg = SecretsCfg()
    modules: ModulesCfg = ModulesCfg()

    class Config:
        extra = "forbid"

    @validator("experiment_name")
    def _nonempty(cls, v):
        assert v and v.strip(), "experiment_name must not be empty"
        return v

    def resolve_run_dirs(self, run_id: str) -> Dict[str, Path]:
        art = Path(self.paths.artifacts_root)
        models_root = art / self.paths.models_subdir
        base = (models_root / run_id).resolve()
        logs = Path(self.paths.logs_root)

        for p in [models_root, base, logs, Path(self.paths.cache_root)]:
            Path(p).mkdir(parents=True, exist_ok=True)

        return {
            "run_dir": base,
            "models_dir": models_root.resolve(),
            "logs_dir": logs.resolve()
        }

# ============================================================
# Utility Loaders
# ============================================================

def load_json(path: Path) -> Dict[str, Any]:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)

def load_global_config(path: Path) -> GlobalConfig:
    with open(path, "r", encoding="utf-8") as f:
        raw = json.load(f)

    base_dir = path.parent.resolve()
    if "paths" in raw:
        for k, v in list(raw["paths"].items()):
            # models_subdir NICHT absolut machen
            if k == "models_subdir":
                continue
            if isinstance(v, str) and not v.startswith("/"):
                raw["paths"][k] = str((base_dir / v).resolve())

    return GlobalConfig(**raw)

def load_module_defaults(global_cfg: GlobalConfig, module_name: str) -> Dict[str, Any]:
    cfg_root = Path(global_cfg.paths.configs_root)
    default_path = cfg_root / f"{module_name}_config.json"
    logger.info(f"[DEBUG] default_path={default_path}")
    if not default_path.exists():
        raise FileNotFoundError(
            f"Default config for module '{module_name}' not found at {default_path}"
        )
    return load_json(default_path)

# ============================================================
# Generic Effective Config Builder
# ============================================================

def build_effective_config(
    global_cfg: GlobalConfig,
    module_name: str,
    *,
    explicit_override_model: Optional[BaseModel] = None,
) -> Dict[str, Any]:
    defaults = load_module_defaults(global_cfg, module_name)
    merged = deepcopy(defaults)

    module_overrides = getattr(global_cfg.modules, module_name, None)
    if module_overrides:
        ov_dict = module_overrides.dict(exclude_unset=True, exclude_none=True)
        for key, val in ov_dict.items():
            if isinstance(val, dict) and key in merged and isinstance(merged[key], dict):
                merged[key].update(val)
            else:
                merged[key] = val

    if explicit_override_model:
        ov_dict = explicit_override_model.dict(exclude_unset=True, exclude_none=True)
        for key, val in ov_dict.items():
            if isinstance(val, dict) and key in merged and isinstance(merged[key], dict):
                merged[key].update(val)
            else:
                merged[key] = val

    return merged

# ============================================================
# Specialized Effective Config Builders
# ============================================================

class EffectiveDataLoaderConfig(BaseModel):
    schema: DataSchemaCfg
    augmentation: DataAugmentCfg
    loader: DataLoaderCfg

def effective_data_loader_config(global_cfg: GlobalConfig) -> EffectiveDataLoaderConfig:
    merged = build_effective_config(global_cfg, "data_loader")
    merged.setdefault("schema", {})
    merged.setdefault("augmentation", {})
    merged.setdefault("loader", {})
    return EffectiveDataLoaderConfig(
        schema=DataSchemaCfg(**merged["schema"]),
        augmentation=DataAugmentCfg(**merged["augmentation"]),
        loader=DataLoaderCfg(**merged["loader"]),
    )

def effective_trainer_config(global_cfg: GlobalConfig) -> TrainerCfg:
    merged = build_effective_config(global_cfg, "trainer")
    return TrainerCfg(**merged)

def effective_logger_config(global_cfg: GlobalConfig) -> Dict[str, Any]:
    return build_effective_config(global_cfg, "logger")

def effective_plotter_config(global_cfg: GlobalConfig) -> Dict[str, Any]:
    return build_effective_config(global_cfg, "plotter")
