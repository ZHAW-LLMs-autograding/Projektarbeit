#!/usr/bin/env python3
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
import re
 
MODEL_DIR = "artifacts/models/huggingface__meta-llama__Meta-Llama-3.1-8B"
 
PROMPT_TEMPLATE = (
    "You are a grading model.\n\n"
    "Question:\n{q}\n\n"
    "Reference answer:\n{r}\n\n"
    "Provided answer (student):\n{p}\n\n"
    "Based on the question, the reference answer and the student's answer, "
    "predict a normalized grade between 0.0 and 1.0 that measures how correct the student's answer is.\n\n"
    "Answer only with a single number between 0.0 and 1.0.\n\n"
    "Grade:"
)
 
 
def build_prompt(q: str, r: str, p: str) -> str:
    return PROMPT_TEMPLATE.format(q=q, r=r, p=p)
 
 
def grade_example(model, tokenizer, q: str, r: str, p: str) -> float:
    """Gibt eine Grade-Prediction des Base-Modells zurück (Baseline)."""
    prompt = build_prompt(q, r, p)
    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
    input_ids = inputs["input_ids"]
 
    with torch.no_grad():
        output_ids = model.generate(
            **inputs,
            max_new_tokens=16,           # wir erwarten nur eine kurze Zahl
            temperature=0.1,             # deterministisch für Baseline
            top_p=1.0,
            repetition_penalty=1.0,
            pad_token_id=tokenizer.pad_token_id,
        )
 
    generated_ids = output_ids[0][input_ids.shape[-1]:]
    raw_output = tokenizer.decode(generated_ids, skip_special_tokens=True).strip()
 
    # Erste Zahl zwischen 0 und 1 extrahieren
    match = re.search(r"\b([01](?:\.\d+)?)\b", raw_output)
    if not match:
        # Fallback: wenn alles schiefgeht, z.B. 0.0 setzen oder NaN zurückgeben
        return 0.0
 
    grade = float(match.group(1))
    # Sicherheitshalber in [0, 1] clampen
    grade = max(0.0, min(1.0, grade))
    return grade
 
 
def main():
    print("Lade Llama-3.1-8B Base...")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_DIR)
 
    # pad_token fix
    if tokenizer.pad_token_id is None:
        if tokenizer.eos_token_id is not None:
            tokenizer.pad_token = tokenizer.eos_token
        else:
            tokenizer.add_special_tokens({"pad_token": "<|pad|>"})
 
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_DIR,
        device_map="auto",
        dtype="auto",
    )
    model.eval()
 
    print("✔ Modell geladen.")
    print("Baseline-Grader (Base-Model). Mit 'exit' bei der Frage-Eingabe beenden.\n")
 
    try:
        while True:
            q = input("❓ Frage: ").strip()
            if q.lower() in {"exit", "quit"}:
                break
            if not q:
                continue
 
            r = input("🏆 Referenzantwort: ").strip()
            p = input("🧑‍🎓 Schülerantwort: ").strip()
            print()
 
            grade = grade_example(model, tokenizer, q, r, p)
 
            print(f"📊 Baseline-Note (0.0–1.0): {grade:.3f}\n")
 
    except (KeyboardInterrupt, EOFError):
        print("\nBeende Session. ✅")
 
 
if __name__ == "__main__":
    main()
