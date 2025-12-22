#!/usr/bin/env python3
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
import re
from pathlib import Path
 
# Ordner, in dem dieses Script liegt
SCRIPT_DIR = Path(__file__).resolve().parent
 
# Lokaler Modellordner für Llama 3.1 Instruct
# → falls dein Ordner anders heißt, HIER anpassen
MODEL_DIR = "artifacts/models/huggingface__meta-llama__Meta-Llama-3.1-8B-Instruct"
 
SYSTEM_MSG = (
    "You are a grading model. Given a question, a reference answer and a student's answer, "
    "you must output only a single number between 0.0 and 1.0 representing how correct "
    "the student's answer is. Do not explain, do not add text, only output the number."
)
 
def build_prompt(q: str, r: str, p: str, tokenizer) -> str:
    """
    Baut einen Chat-Prompt im Llama-3.1-Instruct-Format über apply_chat_template.
    """
    messages = [
        {"role": "system", "content": SYSTEM_MSG},
        {
            "role": "user",
            "content": (
                f"Question:\n{q}\n\n"
                f"Reference answer:\n{r}\n\n"
                f"Provided answer (student):\n{p}\n\n"
                "Return only the grade as a number between 0.0 and 1.0."
            ),
        },
    ]
 
    prompt = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True,  # <assistant> Teil anhängen
    )
    return prompt
 
 
def grade_example(model, tokenizer, q: str, r: str, p: str) -> float:
    """
    Fragt das Instruct-Modell ab und extrahiert eine Note zwischen 0.0 und 1.0.
    """
    prompt = build_prompt(q, r, p, tokenizer)
    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
    input_ids = inputs["input_ids"]
 
    with torch.no_grad():
        output_ids = model.generate(
            **inputs,
            max_new_tokens=16,         # wir erwarten nur eine kurze Zahl
            temperature=0.0001,           # deterministisch
            top_p=1.0,
            repetition_penalty=1.0,
            pad_token_id=tokenizer.pad_token_id,
            eos_token_id=tokenizer.eos_token_id,
        )
 
    # Nur neu generierte Tokens dekodieren
    generated_ids = output_ids[0][input_ids.shape[-1]:]
    raw = tokenizer.decode(generated_ids, skip_special_tokens=True).strip()
    # Debug: falls du sehen willst, was das Modell wirklich ausspuckt
   
    print(f"RAW OUTPUT: {raw!r}")
 
    match = re.search(r"\b([01](?:\.\d+)?)\b", raw)
    if not match:
        # Wenn gar nichts Sinnvolles kommt: 0.0 als Fallback
        return 0.0
 
    grade = float(match.group(1))
    return max(0.0, min(1.0, grade))
 
 
def main():
    print(f"📦 Lade Instruct-Modell aus: {MODEL_DIR}")
 
    if not Path(MODEL_DIR).exists():
        raise FileNotFoundError(f"Modellordner existiert nicht: {MODEL_DIR}")
 
    tokenizer = AutoTokenizer.from_pretrained(str(MODEL_DIR))
 
    # pad_token fixen
    if tokenizer.pad_token_id is None:
        if tokenizer.eos_token_id is not None:
            tokenizer.pad_token = tokenizer.eos_token
        else:
            tokenizer.add_special_tokens({"pad_token": "<|pad|>"})
 
    model = AutoModelForCausalLM.from_pretrained(
        str(MODEL_DIR),
        device_map="auto",
        dtype="auto",
    )
    model.eval()
 
    print("✔ Instruct-Modell geladen.")
    print("🔎 Instruct-Baseline-Grader, 'exit' zum Beenden.\n")
 
    try:
        while True:
            q = input("❓ Frage: ").strip()
            if q.lower() in {"exit", "quit"}:
                break
            if not q:
                continue
 
            r = input("🏆 Referenzantwort: ").strip()
            p = input("🧑‍🎓 Schülerantwort: ").strip()
 
            grade = grade_example(model, tokenizer, q, r, p)
            print(f"\n📊 Note (0.0–1.0): {grade:.3f}\n")
 
    except (KeyboardInterrupt, EOFError):
        print("\nBeende Session. ✅")
 
 
if __name__ == "__main__":
    main()
