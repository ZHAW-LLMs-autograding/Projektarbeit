import os
import numpy as np
import pandas as pd
import torch
from pathlib import Path

from datasets import Dataset, disable_caching
from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
    Trainer,
    TrainingArguments,
    DataCollatorWithPadding,
    set_seed,
)

# -------------------------------------------------------------------
# GLOBAL CONFIG
# -------------------------------------------------------------------
set_seed(42)
disable_caching()

ROOT = Path(__file__).resolve().parents[1]

TRAIN_PATH = ROOT / "trainingsdata" / "ASAG2024" / "student_ranking" / "rank_train.parquet"
VAL_PATH   = ROOT / "trainingsdata" / "ASAG2024" / "student_ranking" / "rank_validation.parquet"

MODEL_NAME = "bert-base-uncased"
MAX_LEN = 256
BATCH_SIZE = 16
LR = 5e-5
EPOCHS = 20

OUT_DIR  = ROOT / "artifacts" / "models" / "bert"
BEST_DIR = OUT_DIR / "best"

OUT_DIR.mkdir(parents=True, exist_ok=True)
BEST_DIR.mkdir(parents=True, exist_ok=True)

# -------------------------------------------------------------------
# PROMPT TEMPLATE (gleich wie bei LLaMA)
# -------------------------------------------------------------------
PROMPT_TEMPLATE = (
    "Question:\n{q}\n\n"
    "Reference answer:\n{r}\n\n"
    "Provided answer:\n{p}\n\n"
    "Predict the grade between 0.0 and 1.0:"
)

def build_prompt(q, r, p):
    return PROMPT_TEMPLATE.format(q=q, r=r, p=p)

# -------------------------------------------------------------------
# Load Dataset
# -------------------------------------------------------------------
def load_dataset():
    train_df = pd.read_parquet(TRAIN_PATH)
    val_df   = pd.read_parquet(VAL_PATH)

    for df in (train_df, val_df):
        df["question"]         = df["question"].fillna("").astype(str)
        df["reference_answer"] = df["reference_answer"].fillna("").astype(str)
        df["provided_answer"]  = df["provided_answer"].fillna("").astype(str)
        df["normalized_grade"] = df["normalized_grade"].astype(float)

    train_prompts = [
        build_prompt(q, r, p)
        for q, r, p in zip(
            train_df["question"],
            train_df["reference_answer"],
            train_df["provided_answer"]
        )
    ]

    val_prompts = [
        build_prompt(q, r, p)
        for q, r, p in zip(
            val_df["question"],
            val_df["reference_answer"],
            val_df["provided_answer"]
        )
    ]

    train_ds = Dataset.from_dict({"text": train_prompts, "labels": train_df["normalized_grade"].tolist()})
    val_ds   = Dataset.from_dict({"text": val_prompts,   "labels": val_df["normalized_grade"].tolist()})

    return train_ds, val_ds

# -------------------------------------------------------------------
# Tokenizer mapping
# -------------------------------------------------------------------
def encode_examples(tokenizer, ds):
    return ds.map(
        lambda batch: tokenizer(batch["text"], truncation=True, max_length=MAX_LEN),
        batched=True,
        remove_columns=["text"]
    )

# -------------------------------------------------------------------
# Metrics (used later in evaluation, NOT in training)
# -------------------------------------------------------------------
def compute_metrics(eval_pred):
    preds = eval_pred.predictions
    if isinstance(preds, tuple):
        preds = preds[0]
    preds = np.squeeze(preds)
    labels = eval_pred.label_ids
    mae = float(np.mean(np.abs(preds - labels)))
    rmse = float(np.sqrt(np.mean((preds - labels)**2)))
    return {"mae": mae, "rmse": rmse}

# -------------------------------------------------------------------
# MAIN
# -------------------------------------------------------------------
def main():
    assert torch.cuda.is_available(), "CUDA GPU required!"

    print(f"[INFO] Using GPU: {torch.cuda.get_device_name(0)}")
    print(f"[INFO] Loading BERT model: {MODEL_NAME}")

    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, use_fast=True)

    print("[INFO] Loading datasets…")
    train_ds, val_ds = load_dataset()

    print("[INFO] Tokenizing…")
    train_tok = encode_examples(tokenizer, train_ds)
    val_tok   = encode_examples(tokenizer, val_ds)

    for d in (train_tok, val_tok):
        cols = ["input_ids", "attention_mask", "labels"]
        if "token_type_ids" in d.features:
            cols.append("token_type_ids")
        d.set_format(type="torch", columns=cols)

    print("[INFO] Preparing regression model…")
    model = AutoModelForSequenceClassification.from_pretrained(
        MODEL_NAME,
        num_labels=1,
        problem_type="regression"
    ).cuda()

    collator = DataCollatorWithPadding(tokenizer)

    # -------------------------------------------------------------------
    # MINIMAL legacy-compatible arguments (your version only supports these)
    # -------------------------------------------------------------------
    args = TrainingArguments(
        output_dir=str(OUT_DIR),
        logging_dir=str(OUT_DIR / "logs"),
        learning_rate=LR,
        per_device_train_batch_size=BATCH_SIZE,
        per_device_eval_batch_size=BATCH_SIZE,
        num_train_epochs=EPOCHS,
        weight_decay=0.0,
        logging_steps=200,
        fp16=True,
    )

    trainer = Trainer(
        model=model,
        args=args,
        train_dataset=train_tok,
        eval_dataset=None,   # << evaluation removed (not supported!)
        data_collator=collator,
        tokenizer=tokenizer,
    )

    print("[INFO] Start training (NO evaluation during training)…")
    trainer.train()

    print(f"[INFO] Saving model → {BEST_DIR}")
    trainer.save_model(str(BEST_DIR))
    tokenizer.save_pretrained(str(BEST_DIR))

    print("[OK] Training FINISHED")

if __name__ == "__main__":
    main()
