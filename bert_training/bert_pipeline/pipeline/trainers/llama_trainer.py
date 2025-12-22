from __future__ import annotations

import math
from typing import Dict, List, Any
from contextlib import nullcontext

import torch
from torch import nn
from tqdm import tqdm

from .base_trainer import BaseTrainer

from peft import LoraConfig, get_peft_model
try:
    from peft.utils.other import prepare_model_for_kbit_training
except Exception:
    prepare_model_for_kbit_training = None

import bitsandbytes as bnb

PROMPT_TEMPLATE = (
    "[INST]\n"
    "Question:\n{q}\n\n"
    "Reference answer:\n{r}\n\n"
    "Provided answer:\n{p}\n"
    "[/INST]\n"
    "Predict the grade based on the given question with the reference answer and the provided answer made by the student and return a normalized grade between 0.0 and 1.0."
)

def _build_texts(batch: Dict[str, Any]) -> list[str]:
    questions = batch["question"]
    ref_answ = batch["reference_answer"]
    prov_answ = batch["provided_answer"]
    if isinstance(questions, (list, tuple)):
        return [PROMPT_TEMPLATE.format(q=q, r=r, p=p) for q, r, p in zip(questions, ref_answ, prov_answ)]
    return [PROMPT_TEMPLATE.format(q=questions, r=ref_answ, p=prov_answ)]

def _to_tensor(x, device, dtype=torch.float32):
    if isinstance(x, torch.Tensor):
        return x.to(device=device, dtype=dtype)
    return torch.tensor(x, device=device, dtype=dtype)

def _coalesce(val, default):
    return default if val is None else val

class LlamaTrainer(BaseTrainer):
    """
    Llama 3.1 Regression-Trainer mit LoRA.
    - Ungewichteter Loss (SmoothL1/MSE)
    - Separater Regressionskopf [B,1]
    - Stabilität: Warmup, Grad-Clipping, LoRA-Warmup
    - Zwei Optimizer: LoRA=bnb Adam8bit, Head=AdamW (FP32)
    - FIX: Nur LoRA-Adapter werden (ent)frozen → keine int8-Param-Fehler
    """

    def __init__(self, cfg, model, datamodule, logger=None):
        super().__init__(cfg, model, datamodule, logger)

        self.device = torch.device(_coalesce(self.cfg.get("device"), ("cuda" if torch.cuda.is_available() else "cpu")))
        self.max_length = int(_coalesce(self.cfg.get("max_length"), 512))
        self.mixed_precision = bool(_coalesce(self.cfg.get("mixed_precision"), True))
        self.grad_accum_steps = int(_coalesce(self.cfg.get("gradient_accumulation_steps"), 1))
        self.lr = float(_coalesce(self.cfg.get("lr"), 2e-5))          # LoRA LR
        self.lr_head = float(_coalesce(self.cfg.get("lr_head"), 3e-4))# Head LR
        self.epochs = int(_coalesce(self.cfg.get("epochs"), 2))
        self.weight_decay = float(_coalesce(self.cfg.get("weight_decay"), 0.0))
        self.max_grad_norm = float(_coalesce(self.cfg.get("max_grad_norm"), 0.5))
        self.label_field = _coalesce(self.cfg.get("label_field"), "normalized_grade")
        self.use_smooth_l1 = bool(_coalesce(self.cfg.get("use_smooth_l1"), True))
        self.smooth_l1_beta = float(_coalesce(self.cfg.get("smooth_l1_beta"), 0.5))
        self.report_clamp_eval = bool(_coalesce(self.cfg.get("report_clamp_eval"), True))
        self.warmup_ratio = float(_coalesce(self.cfg.get("warmup_ratio"), 0.10))               # 10% linear warmup
        self.freeze_lora_warmup_steps = int(_coalesce(self.cfg.get("freeze_lora_warmup_steps"), 300))

        if self.use_smooth_l1:
            self.criterion = nn.SmoothL1Loss(beta=self.smooth_l1_beta, reduction="mean")
        else:
            self.criterion = nn.MSELoss(reduction="mean")

        self._input_device = self._infer_input_device()

        # Tokenizer / Padding
        try:
            tok = self.model.tokenizer
            mdl = self.model.model
            if getattr(tok, "pad_token", None) is None:
                tok.pad_token = tok.eos_token
            tok.padding_side = "right"
            if hasattr(mdl, "config"):
                mdl.config.pad_token_id = tok.pad_token_id
                if hasattr(mdl.config, "output_hidden_states"):
                    mdl.config.output_hidden_states = True
        except Exception:
            pass

        # TF32 & Cache
        try:
            if hasattr(self.model.model, "config") and hasattr(self.model.model.config, "use_cache"):
                self.model.model.config.use_cache = False
            torch.backends.cuda.matmul.allow_tf32 = True
        except Exception:
            pass

        # LoRA
        self._has_meta = self._model_has_meta_params()
        if not self._has_meta:
            try:
                if hasattr(self.model.model, "gradient_checkpointing_enable"):
                    self.model.model.gradient_checkpointing_enable()
            except Exception:
                pass
        self._enable_lora()

        # AMP
        use_cuda = (self._input_device.type == "cuda")
        bf16_supported = use_cuda and torch.cuda.is_bf16_supported()
        self._use_autocast = self.mixed_precision and use_cuda
        self._amp_dtype = torch.bfloat16 if bf16_supported else torch.float16
        self.scaler = torch.cuda.amp.GradScaler() if (self._use_autocast and self._amp_dtype == torch.float16) else None

        # Regression-Head
        hidden_size = getattr(getattr(self.model.model, "config", None), "hidden_size", None)
        if hidden_size is None:
            hidden_size = getattr(getattr(self.model.model, "config", None), "hidden_dim", None)
        if hidden_size is None:
            raise ValueError("Could not infer hidden_size from model.config")

        self.reg_head = nn.Linear(hidden_size, 1)
        nn.init.xavier_uniform_(self.reg_head.weight)
        with torch.no_grad():
            self.reg_head.bias.fill_(0.5)
        self.reg_head.to(self._input_device)

    # ---------- intern ----------

    def _model_has_meta_params(self) -> bool:
        for p in self.model.model.parameters():
            if getattr(p.device, "type", None) == "meta":
                return True
        return False

    def _enable_lora(self):
        base = self.model.model
        try:
            if prepare_model_for_kbit_training is not None:
                base = prepare_model_for_kbit_training(base, use_gradient_checkpointing=not self._has_meta)
        except Exception as e:
            print(f"[WARN] prepare_model_for_kbit_training failed: {e}")

        cfg = LoraConfig(
            r=8,
            lora_alpha=16,
            target_modules=["q_proj", "k_proj", "v_proj", "o_proj"],
            lora_dropout=0.05,
            bias="none",
            task_type="SEQ_CLS",
        )
        try:
            self.model.model = get_peft_model(base, cfg)
            self.model.model.print_trainable_parameters()
        except Exception as e:
            print(f"[WARN] LoRA setup failed: {e}")

    def _infer_input_device(self):
        try:
            return self.model.model.get_input_embeddings().weight.device
        except Exception:
            return torch.device("cuda" if torch.cuda.is_available() else "cpu")

    def _amp_context(self):
        if not self._use_autocast:
            return nullcontext()
        try:
            return torch.autocast("cuda", dtype=self._amp_dtype)
        except Exception:
            from torch.cuda.amp import autocast
            return autocast(dtype=self._amp_dtype)

    def _encode_batch(self, texts: list[str]) -> Dict[str, torch.Tensor]:
        tok = self.model.tokenizer(
            texts,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=self.max_length,
        )
        dev = self._input_device
        return {k: v.to(dev) for k, v in tok.items()}

    # ---------- LoRA Toggle Helpers ----------

    @staticmethod
    def _is_lora_param(name: str) -> bool:
        # gängige PEFT-Namen enthalten lora_, lora_A, lora_B, lora_dropout
        n = name.lower()
        return ("lora_" in n) or ("lora_a" in n) or ("lora_b" in n)

    def _toggle_lora_trainable(self, enabled: bool):
        """Nur LoRA-Parameter (float) (de-)aktivieren; int8-Grundgewichte nicht anfassen."""
        for name, p in self.model.model.named_parameters():
            if self._is_lora_param(name) and p.is_floating_point():
                p.requires_grad_(enabled)

    # ---------- core ----------

    def _forward_regression(self, inputs: Dict[str, torch.Tensor]) -> torch.Tensor:
        outputs = self.model.model(**inputs, output_hidden_states=True, return_dict=True)
        if hasattr(outputs, "last_hidden_state") and outputs.last_hidden_state is not None:
            last_hidden = outputs.last_hidden_state
        elif hasattr(outputs, "hidden_states") and outputs.hidden_states is not None:
            last_hidden = outputs.hidden_states[-1]
        else:
            raise RuntimeError("Model outputs did not contain hidden states.")

        attn = inputs.get("attention_mask", None)
        if attn is None:
            idx = last_hidden.new_full((last_hidden.size(0),), last_hidden.size(1) - 1, dtype=torch.long)
        else:
            lengths = attn.long().sum(dim=1)
            idx = lengths - 1

        batch_idx = torch.arange(last_hidden.size(0), device=last_hidden.device)
        pooled = last_hidden[batch_idx, idx]
        preds = self.reg_head(pooled)
        return preds

    def compute_loss_and_preds(self, batch: Dict[str, Any]):
        texts = _build_texts(batch)
        inputs = self._encode_batch(texts)
        labels = _to_tensor(batch[self.label_field], self._input_device, dtype=torch.float32).view(-1, 1)

        with self._amp_context():
            preds = self._forward_regression(inputs)
            loss = self.criterion(preds, labels)

        return loss, preds.detach(), labels.detach()

    def compute_loss(self, batch):
        loss, _, _ = self.compute_loss_and_preds(batch)
        return loss

    # ---------- optim/sched ----------

    def _build_optimizers(self):
        # LoRA mit bnb Adam8bit (nur die aktuell requires_grad=True Parameter)
        lora_params = [p for n, p in self.model.model.named_parameters() if p.requires_grad]
        opt_lora = bnb.optim.Adam8bit([{"params": lora_params, "lr": self.lr, "weight_decay": self.weight_decay}])

        # Head separat in AdamW (FP32)
        opt_head = torch.optim.AdamW(self.reg_head.parameters(), lr=self.lr_head, weight_decay=0.0)
        return opt_lora, opt_head

    def _build_linear_warmup(self, optimizer, steps_per_epoch: int):
        warmup_steps = max(1, int(self.warmup_ratio * steps_per_epoch))
        def lr_lambda(current_step):
            if current_step < warmup_steps:
                return float(current_step + 1) / float(max(1, warmup_steps))
            return 1.0
        return torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=lr_lambda)

    # ---------- training ----------

    def train_epoch(self) -> float:
        self.model.model.train()
        self.reg_head.train()

        loader = self.dm.train_loader()
        steps_per_epoch = len(loader)

        # LoRA zunächst einfrieren (nur die Adapter!)
        lora_frozen = True if self.freeze_lora_warmup_steps > 0 else False
        if lora_frozen:
            self._toggle_lora_trainable(False)  # nur LoRA-Adapter einfrieren
        # Regressionskopf trainierbar
        for p in self.reg_head.parameters():
            p.requires_grad_(True)

        opt_lora, opt_head = self._build_optimizers()
        sch_lora = self._build_linear_warmup(opt_lora, steps_per_epoch)
        sch_head = self._build_linear_warmup(opt_head, steps_per_epoch)

        total_loss = 0.0
        step = 0
        opt_lora.zero_grad(set_to_none=True)
        opt_head.zero_grad(set_to_none=True)

        pbar = tqdm(loader, desc="Training", dynamic_ncols=True)
        for batch in pbar:
            loss, preds, labels = self.compute_loss_and_preds(batch)

            # Debug (inkl. hübsch formatierte pred_labels)
            try:
                preds_cpu = preds.view(-1).float().cpu()
                labels_cpu = labels.view(-1).float().cpu()
                p_min, p_max, p_mean = float(preds_cpu.min()), float(preds_cpu.max()), float(preds_cpu.mean())
                head_weight_norm = float(self.reg_head.weight.data.norm().detach().cpu())
                head_bias = float(self.reg_head.bias.data.mean().detach().cpu())
                show_k = min(4, preds_cpu.numel())
                pred_labels_fmt = "[" + ", ".join(f"{v:.2f}" for v in preds_cpu.tolist()[:show_k]) + "]"
                labels_fmt = "[" + ", ".join(f"{v:.2f}" for v in labels_cpu.tolist()[:show_k]) + "]"
                pbar.write(
                    f"\n--- Batch Debug (step {step}) ---\n"
                    f"Loss: {loss.item():.6f}\n"
                    f"Preds stats: min={p_min:.3f}, max={p_max:.3f}, mean={p_mean:.3f}\n"
                    f"pred_labels: {pred_labels_fmt}\n"
                    f"Labels[:{show_k}]: {labels_fmt}\n"
                    f"Head norm={head_weight_norm:.3f}, bias={head_bias:.3f}\n"
                    f"-------------------------------\n"
                )
            except Exception:
                pass

            # Backward
            if self.scaler is not None:
                self.scaler.scale(loss / self.grad_accum_steps).backward()
            else:
                (loss / self.grad_accum_steps).backward()

            # optional: Grad-Inspect
            try:
                gn_w = float(self.reg_head.weight.grad.detach().data.norm().cpu()) if self.reg_head.weight.grad is not None else 0.0
                gb = float(self.reg_head.bias.grad.detach().data.mean().cpu()) if self.reg_head.bias.grad is not None else 0.0
                pbar.write(f"[GRAD] head ||W||_grad={gn_w:.6f}, bias_grad_mean={gb:.6f}")
            except Exception:
                pass

            if (step + 1) % self.grad_accum_steps == 0:
                # unscale + clip
                if self.scaler is not None:
                    self.scaler.unscale_(opt_head)
                    self.scaler.unscale_(opt_lora)
                torch.nn.utils.clip_grad_norm_(self.reg_head.parameters(), self.max_grad_norm)
                if not lora_frozen:
                    # nur LoRA-Adapter haben requires_grad=True → ok
                    torch.nn.utils.clip_grad_norm_([p for n,p in self.model.model.named_parameters() if p.requires_grad], self.max_grad_norm)

                # step beide Optimizer
                if self.scaler is not None:
                    self.scaler.step(opt_head)
                    if not lora_frozen:
                        self.scaler.step(opt_lora)
                    self.scaler.update()
                else:
                    opt_head.step()
                    if not lora_frozen:
                        opt_lora.step()

                opt_head.zero_grad(set_to_none=True)
                if not lora_frozen:
                    opt_lora.zero_grad(set_to_none=True)

                sch_head.step()
                if not lora_frozen:
                    sch_lora.step()

                # LoRA nach Warmup freigeben (nur Adapter toggeln!)
                if lora_frozen and step + 1 >= self.freeze_lora_warmup_steps:
                    self._toggle_lora_trainable(True)
                    lora_frozen = False
                    # Optimizer erneut bauen, damit LoRA-Params sicher enthalten sind
                    opt_lora, _ = self._build_optimizers()
                    sch_lora = self._build_linear_warmup(opt_lora, steps_per_epoch)

            total_loss += loss.item()
            step += 1

        return total_loss / max(1, len(loader))

    # ---------- evaluation ----------

    @torch.no_grad()
    def evaluate(self) -> Dict[str, float]:
        self.model.model.eval()
        self.reg_head.eval()
        loader = self.dm.val_loader()

        losses, all_preds, all_labels = [], [], []
        pbar = tqdm(loader, desc="Validation", dynamic_ncols=True)
        for batch in pbar:
            loss, preds, labels = self.compute_loss_and_preds(batch)
            losses.append(loss.item())
            if self.report_clamp_eval:
                preds = preds.clamp(0.0, 1.0)
            all_preds.append(preds.float().cpu())
            all_labels.append(labels.float().cpu())

        if not all_preds:
            return {"val_loss": float("nan"), "rmse": float("nan"), "mae": float("nan"), "r2": float("nan")}

        preds = torch.cat(all_preds, dim=0).squeeze(-1)
        labels = torch.cat(all_labels, dim=0).squeeze(-1)

        mse = torch.mean((preds - labels) ** 2).item()
        rmse = math.sqrt(mse)
        mae = torch.mean(torch.abs(preds - labels)).item()
        r2 = self._r2_score(labels.numpy(), preds.numpy())
        return {"val_loss": sum(losses) / len(losses), "rmse": rmse, "mae": mae, "r2": r2}

    @staticmethod
    def _r2_score(y_true, y_pred):
        y_true_mean = y_true.mean()
        ss_res = ((y_true - y_pred) ** 2).sum()
        ss_tot = ((y_true - y_true_mean) ** 2).sum()
        if ss_tot == 0:
            return float("nan")
        return 1.0 - (ss_res / ss_tot)
