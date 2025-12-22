import json
import time
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import PeftModel
import pandas as pd
from tqdm import tqdm

# -------------------------------------------------------------
# CONFIG
# -------------------------------------------------------------
BASE_MODEL = "/home/sercan/asag_pipeline/pipeline/artifacts/models/kevin_llama3_base"
LORA_MODEL = "/home/sercan/asag_pipeline/pipeline/artifacts/models/llama3.1_base_finetuned"
CKPT_PATH  = "/home/sercan/asag_pipeline/pipeline/artifacts/models/checkpoints/model_epoch18_base.pt"
TEST_DATA  = "/home/sercan/asag_pipeline/pipeline/trainingsdata/ASAG2024/student_ranking/rank_test.parquet"

OUT_PRED  = "/home/sercan/asag_pipeline/pipeline/artifacts/eval_test/predictions_llama3_base_lora.json"
OUT_METS  = "/home/sercan/asag_pipeline/pipeline/artifacts/eval_test/metrics_llama3_base_lora.json"

PROMPT_TEMPLATE = (
    "Question:\n{q}\n\n"
    "Reference answer:\n{r}\n\n"
    "Provided answer:\n{p}\n"
    "Predict the grade based on the given question with the reference answer "
    "and the provided answer made by the student and return a normalized grade between 0.0 and 1.0."
)


# -------------------------------------------------------------
# LOAD MODELS
# -------------------------------------------------------------
print("[INFO] Loading base model…")
tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL)
base = AutoModelForCausalLM.from_pretrained(
    BASE_MODEL,
    torch_dtype=torch.float16,
    device_map="cuda",
)

print("[INFO] Loading LoRA adapter…")
model = PeftModel.from_pretrained(base, LORA_MODEL)

print("[INFO] Loading checkpoint (regression head)…")
ckpt = torch.load(CKPT_PATH, map_location="cpu")
reg_head_state = ckpt["reg_head"]

# build regression head
hidden = base.config.hidden_size
reg_head = torch.nn.Linear(hidden, 1)
reg_head.load_state_dict(reg_head_state, strict=True)
reg_head = reg_head.to("cuda")
reg_head.eval()

model.eval()


# -------------------------------------------------------------
# HELPER: RUN REGRESSION FOR ONE SAMPLE
# -------------------------------------------------------------
@torch.no_grad()
def predict_grade(prompt: str):
    inputs = tokenizer(prompt, return_tensors="pt", padding=True, truncation=True).to("cuda")

    # forward with hidden states
    out = model.model(
        **inputs,
        output_hidden_states=True,
        return_dict=True
    )
    hidden_states = out.hidden_states[-1]       # last layer: [B, seq, hidden]
    attn = inputs["attention_mask"]             # lengths
    lengths = attn.sum(dim=1)
    last_token_index = lengths - 1

    batch_idx = torch.arange(hidden_states.size(0), device="cuda")
    pooled = hidden_states[batch_idx, last_token_index]

    pred = reg_head(pooled).squeeze().item()
    return float(pred)   # no clamping



# -------------------------------------------------------------
# RUN EVALUATION
# -------------------------------------------------------------
df = pd.read_parquet(TEST_DATA)
results = []

print("[INFO] Evaluating…")
start_all = time.time()

for i, row in tqdm(df.iterrows(), total=len(df)):
    q = row["question"]
    r = row["reference_answer"]
    p = row["provided_answer"]

    prompt = PROMPT_TEMPLATE.format(q=q, r=r, p=p)

    t0 = time.time()
    pred = predict_grade(prompt)
    t1 = time.time()

    results.append({
        "question": q,
        "reference_answer": r,
        "provided_answer": p,
        "y_true": float(row["normalized_grade"]),
        "y_pred": pred,
        "source": row["data_source"],
        "inference_time_ms": (t1 - t0) * 1000
    })


# -------------------------------------------------------------
# METRICS
# -------------------------------------------------------------
y_true = [r["y_true"] for r in results]
y_pred = [r["y_pred"] for r in results]

import numpy as np

MAE = float(np.mean(np.abs(np.array(y_pred) - np.array(y_true))))
RMSE = float(np.sqrt(np.mean((np.array(y_pred) - np.array(y_true)) ** 2)))

w = df["weight"].to_numpy()
w = w / np.mean(w)
W_RMSE = float(np.sqrt(np.average((np.array(y_pred) - np.array(y_true)) ** 2, weights=w)))

metrics = {
    "MAE": MAE,
    "RMSE": RMSE,
    "wRMSE": W_RMSE,
    "num_samples_total": len(df),
    "total_inference_time_s": time.time() - start_all,
    "avg_inference_time_ms": float(np.mean([r["inference_time_ms"] for r in results]))
}

# -------------------------------------------------------------
# SAVE OUTPUT
# -------------------------------------------------------------
with open(OUT_PRED, "w") as f:
    json.dump(results, f, indent=2)

with open(OUT_METS, "w") as f:
    json.dump(metrics, f, indent=2)

print("[DONE] Saved predictions and metrics.")
