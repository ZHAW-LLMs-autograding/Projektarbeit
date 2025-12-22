import json
import time
import math
from pathlib import Path

import torch
from torch import nn
import pandas as pd
from tqdm import tqdm

from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
from peft import LoraConfig, get_peft_model

# -------------------------------------------------------
# Pfade
# -------------------------------------------------------
ROOT = Path(__file__).resolve().parents[1]

BASE_MODEL_PATH = ROOT / "artifacts" / "models" / "huggingface__meta-llama__Meta-Llama-3.1-8B"
CKPT_PATH = ROOT / "artifacts" / "models" / "checkpoints" / "model_epoch18_base.pt"

TEST_PARQUET = (
    ROOT
    / "trainingsdata"
    / "ASAG2024"
    / "student_ranking"
    / "rank_test.parquet"
)

OUT_DIR = ROOT / "metriken" / "benchmarks" / "llama3.1_base_lora"
OUT_DIR.mkdir(parents=True, exist_ok=True)

PREDICTIONS_PATH = OUT_DIR / "predictions.json"
METRICS_PATH = OUT_DIR / "metrics.json"

# -------------------------------------------------------
# Prompt exakt wie im Training (LlamaTrainer.PROMPT_TEMPLATE)
# -------------------------------------------------------
PROMPT_TEMPLATE = (
    "Question:\n{q}\n\n"
    "Reference answer:\n{r}\n\n"
    "Provided answer:\n{p}\n"
    "Predict the grade based on the given question with the reference answer and the provided answer made by the student and return a normalized grade between 0.0 and 1.0."
)


def build_prompt(question: str, reference_answer: str, provided_answer: str) -> str:
    return PROMPT_TEMPLATE.format(q=question, r=reference_answer, p=provided_answer)


# -------------------------------------------------------
# Source-Gewichte für wRMSE (falls du echte Gewichte hast → hier anpassen)
# -------------------------------------------------------
SOURCE_WEIGHTS = {
    "CSD": 1.0,
    "SAF": 1.0,
    "BeSt": 1.0,
    "Stroud": 1.0,
    "SciEnt": 1.0,
}


# -------------------------------------------------------
# Modell + LoRA + Regressionskopf laden
# (quantisiertes NF4, wie im Training)
# -------------------------------------------------------
def load_model():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    print(f"[INFO] Base model: {BASE_MODEL_PATH}")
    print(f"[INFO] Checkpoint: {CKPT_PATH}")

    # Tokenizer
    print("[INFO] Loading tokenizer…")
    tokenizer = AutoTokenizer.from_pretrained(str(BASE_MODEL_PATH))
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "right"

    # Quantisierte Basis (NF4)
    print("[INFO] Loading quantized base model (NF4 via BitsAndBytesConfig)…")
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_compute_dtype=torch.bfloat16,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_use_double_quant=True,
    )

    base = AutoModelForCausalLM.from_pretrained(
        str(BASE_MODEL_PATH),
        quantization_config=bnb_config,
        device_map="cuda",
    )

    # LoRA-Struktur wie im Training
    lora_cfg = LoraConfig(
        r=8,
        lora_alpha=16,
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj"],
        lora_dropout=0.05,
        bias="none",
        task_type="SEQ_CLS",
    )
    model = get_peft_model(base, lora_cfg)

    # Checkpoint laden
    print("[INFO] Loading checkpoint…")
    ckpt = torch.load(str(CKPT_PATH), map_location="cpu")

    model_state = ckpt.get("model", ckpt)
    missing, unexpected = model.load_state_dict(model_state, strict=False)
    print(f"[INFO] Loaded model state_dict (missing={len(missing)}, unexpected={len(unexpected)})")

    model.eval()

    # Regression-Head laden
    print("[INFO] Loading regression head…")
    hidden_size = getattr(model.config, "hidden_size", None)
    if hidden_size is None:
        raise ValueError("Could not infer hidden_size from model.config")

    # Regressionskopf in bf16, damit Dtype zu NF4-Compute passt
    reg_head = nn.Linear(hidden_size, 1, dtype=torch.bfloat16)
    reg_head.load_state_dict(ckpt["reg_head"], strict=False)
    reg_head.to(device)
    reg_head.eval()

    return model, tokenizer, reg_head, device


# -------------------------------------------------------
# Single Prediction (keine Generierung, nur Regression)
# -------------------------------------------------------
@torch.no_grad()
def predict_single(model, tokenizer, reg_head, device, question, reference_answer, provided_answer):
    prompt = build_prompt(question, reference_answer, provided_answer)

    inputs = tokenizer(
        prompt,
        return_tensors="pt",
        padding=False,
        truncation=True,
        max_length=512,
    ).to(device)

    # wie im Trainer: wir arbeiten über hidden_states[-1]
    outputs = model(**inputs, output_hidden_states=True, return_dict=True)

    if not hasattr(outputs, "hidden_states") or outputs.hidden_states is None:
        raise RuntimeError("hidden_states missing in model output")

    last_hidden = outputs.hidden_states[-1]

    attn = inputs.get("attention_mask", None)
    if attn is None:
        idx = last_hidden.new_full(
            (last_hidden.size(0),),
            last_hidden.size(1) - 1,
            dtype=torch.long,
        )
    else:
        lengths = attn.long().sum(dim=1)
        idx = lengths - 1

    batch_idx = torch.arange(last_hidden.size(0), device=last_hidden.device)
    pooled = last_hidden[batch_idx, idx]

    # Dtype an Regressionskopf anpassen (bf16)
    pooled = pooled.to(reg_head.weight.dtype)

    pred = reg_head(pooled)  # [B, 1]
    pred = float(pred.view(-1).item())
    return pred, prompt


# -------------------------------------------------------
# Hauptfunktion: Testdaten laden, auswerten, JSON schreiben
# -------------------------------------------------------
def main():
    print(f"[INFO] Loading test set: {TEST_PARQUET}")
    df = pd.read_parquet(TEST_PARQUET)

    # Erwartete Spaltennamen:
    # question, reference_answer, provided_answer, normalized_grade, (optional) source
    required_cols = {"question", "reference_answer", "provided_answer", "normalized_grade"}
    missing = required_cols - set(df.columns)
    if missing:
        raise ValueError(f"Missing required columns in test parquet: {missing}")

    model, tokenizer, reg_head, device = load_model()

    predictions = []

    total_abs_err = 0.0
    total_sq_err = 0.0
    total_w_sq_err = 0.0
    total_weight = 0.0

    num_total = 0
    num_valid = 0
    total_time_s = 0.0

    print("[INFO] Starting evaluation on TEST data…")
    for row in tqdm(
        df.itertuples(index=False),
        total=len(df),
        desc="Evaluating",
        dynamic_ncols=True,
    ):
        question = getattr(row, "question")
        reference_answer = getattr(row, "reference_answer")
        provided_answer = getattr(row, "provided_answer")
        y_true = float(getattr(row, "normalized_grade"))
        # source ist optional – wenn Spalte fehlt, default "unknown"
        source = getattr(row, "source", "unknown")

        t0 = time.perf_counter()
        try:
            y_pred, raw_prompt = predict_single(
                model, tokenizer, reg_head, device,
                question, reference_answer, provided_answer
            )
        except Exception as e:
            print(f"[WARN] Prediction failed for sample {num_total}: {e}")
            y_pred = None
            raw_text = ""
        else:
            # raw_output_text: wir loggen hier einfach die numerische Vorhersage als String
            raw_text = f"{y_pred:.6f}"
        t1 = time.perf_counter()

        inf_ms = (t1 - t0) * 1000.0
        total_time_s += (t1 - t0)

        entry = {
            "question": question,
            "reference_answer": reference_answer,
            "provided_answer": provided_answer,
            "y_true": y_true,
            "y_pred": y_pred,
            "source": source,
            "raw_output_text": raw_text,
            "inference_time_ms": inf_ms,
        }
        predictions.append(entry)

        num_total += 1

        if y_pred is not None and not (math.isnan(y_pred) or math.isnan(y_true)):
            num_valid += 1
            err = y_pred - y_true
            total_abs_err += abs(err)
            total_sq_err += err * err

            w = SOURCE_WEIGHTS.get(source, 1.0)
            total_w_sq_err += w * err * err
            total_weight += w

    # Metriken
    if num_valid > 0:
        mae = total_abs_err / num_valid
        rmse = math.sqrt(total_sq_err / num_valid)
        if total_weight > 0.0:
            w_rmse = math.sqrt(total_w_sq_err / total_weight)
        else:
            w_rmse = float("nan")
        avg_ms = (total_time_s / num_valid) * 1000.0
    else:
        mae = rmse = w_rmse = avg_ms = float("nan")

    metrics = {
        "MAE": mae,
        "RMSE": rmse,
        "wRMSE": w_rmse,
        "num_samples_total": int(num_total),
        "num_valid_predictions": int(num_valid),
        "total_inference_time_s": total_time_s,
        "avg_inference_time_ms": avg_ms,
    }

    # JSON schreiben
    with open(PREDICTIONS_PATH, "w", encoding="utf-8") as f:
        json.dump(predictions, f, ensure_ascii=False, indent=2)

    with open(METRICS_PATH, "w", encoding="utf-8") as f:
        json.dump(metrics, f, ensure_ascii=False, indent=2)

    print("\n[INFO] Saved:")
    print(f"  - predictions: {PREDICTIONS_PATH}")
    print(f"  - metrics    : {METRICS_PATH}")
    print("\n[INFO] Metrics:")
    for k, v in metrics.items():
        print(f"  {k}: {v}")


if __name__ == "__main__":
    main()
