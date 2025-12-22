import json
import torch
from pathlib import Path
from transformers import AutoTokenizer, AutoModelForSequenceClassification

# --------------------------------------------------------------------
# KONFIGURATION
# --------------------------------------------------------------------
BEST_DIR = Path("/home/sercan/asag_pipeline/pipeline/artifacts/models/bert/checkpoint-17500")  # <--- HIER ANPASSEN!

OUTPUT_FILE = Path("/home/sercan/asag_pipeline/pipeline/utils/BERT-Leakage-Test/results_unseen.json")
INPUT_FILE = Path("/home/sercan/asag_pipeline/pipeline/utils/BERT-Leakage-Test/tests_unseen.json")

PROMPT_TEMPLATE = (
    "Question:\n{q}\n\n"
    "Reference answer:\n{r}\n\n"
    "Provided answer:\n{a}\n\n"
    "Predict the grade between 0.0 and 1.0:"
)

# --------------------------------------------------------------------
# Hilfsfunktion: Prompt bauen
# --------------------------------------------------------------------
def make_prompt(question, reference, answer):
    return PROMPT_TEMPLATE.format(q=question, r=reference, a=answer)

# --------------------------------------------------------------------
# Modell laden
# --------------------------------------------------------------------
print("[INFO] Loading BERT model…")
tokenizer = AutoTokenizer.from_pretrained(str(BEST_DIR))
model = AutoModelForSequenceClassification.from_pretrained(str(BEST_DIR))
model.eval()
model.to("cuda" if torch.cuda.is_available() else "cpu")

device = next(model.parameters()).device

# --------------------------------------------------------------------
# Testdaten laden
# --------------------------------------------------------------------
print("[INFO] Loading test cases…")
with open(INPUT_FILE, "r", encoding="utf-8") as f:
    tests = json.load(f)

results = []

# --------------------------------------------------------------------
# Testfälle auswerten
# --------------------------------------------------------------------
print("[INFO] Running BERT predictions…")

for case in tests:

    prompt = make_prompt(case["question"], case["reference"], case["answer"])

    tokens = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=256).to(device)

    with torch.no_grad():
        logits = model(**tokens).logits.squeeze().item()

    pred = float(logits)

    results.append({
        "id": case["id"],
        "question": case["question"],
        "reference": case["reference"],
        "answer": case["answer"],
        "prediction": pred
    })

    print(f"[OK] {case['id']} → {pred:.3f}")

# --------------------------------------------------------------------
# Ergebnisse speichern
# --------------------------------------------------------------------
print(f"[INFO] Saving results → {OUTPUT_FILE}")
with open(OUTPUT_FILE, "w", encoding="utf-8") as f:
    json.dump(results, f, indent=4)

print("[DONE] All tests evaluated.")
