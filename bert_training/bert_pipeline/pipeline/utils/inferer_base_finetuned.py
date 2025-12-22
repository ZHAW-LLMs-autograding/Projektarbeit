import torch
import time
import re
from transformers import AutoTokenizer, AutoModelForCausalLM

class LlamaBaseFinetunedInferer:
    """
    Wissenschaftlich korrekte Auswertung eines feinabgestimmten Base-LLaMA-Models.
    Nutzt keinen generate()-Call, sondern den Forward-Pass (Regressionskopf).
    Prompt bleibt exakt wie im Training.
    """

    def __init__(self, base_model_path, finetuned_ckpt_path):
        print(f"[INFO] Loading base model: {base_model_path}")
        print(f"[INFO] Loading finetuned weights: {finetuned_ckpt_path}")

        self.tokenizer = AutoTokenizer.from_pretrained(base_model_path)

        # Base-Modell laden
        self.model = AutoModelForCausalLM.from_pretrained(
            base_model_path,
            torch_dtype=torch.float16,
            device_map={"": "cuda"},
        )

        # State-Dict laden (komplettes Fine-Tuned Modell)
        state = torch.load(finetuned_ckpt_path, map_location="cuda")
        model_state = state["model"] if "model" in state else state

        missing, unexpected = self.model.load_state_dict(model_state, strict=False)
        print("[INFO] Missing keys:", missing)
        print("[INFO] Unexpected keys:", unexpected)

        self.model.eval()

    # TRAININGSPROMPT 1:1
    def build_prompt(self, question, reference, answer):
        return f"""
[INST]
Question:
{question}

Reference answer:
{reference}

Provided answer:
{answer}
[/INST]

Predict the grade based on the given question with the reference answer
and the provided answer made by the student and return a normalized
grade between 0.0 and 1.0.
""".strip()

    # extrahiere Zahlen — KEIN Clamping
    def extract_number(self, text):
        nums = re.findall(r"[-+]?\d*\.\d+|\d+", text)
        for n in nums:
            try:
                return float(n)
            except:
                pass
        return None

    def infer(self, question, reference, answer):
        prompt = self.build_prompt(question, reference, answer)
        inputs = self.tokenizer(prompt, return_tensors="pt").to("cuda")

        start = time.time()

        # ✔ Wissenschaftlich korrekt: Forward-Pass statt generate()
        with torch.no_grad():
            out = self.model(**inputs)
            # logits-Shape: [batch, seq_len, vocab]
            # Wir nehmen das letzte Token → das lernte im Training den Score
            last_logits = out.logits[:, -1, :]

        duration_ms = (time.time() - start) * 1000.0

        # logits → Text → Zahl
        text_pred = self.tokenizer.decode(last_logits.argmax(dim=-1))
        pred = self.extract_number(text_pred)

        return text_pred, pred, duration_ms
