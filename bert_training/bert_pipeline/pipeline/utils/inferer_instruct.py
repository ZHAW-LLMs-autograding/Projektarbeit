import torch
import time
import re
from transformers import AutoTokenizer, AutoModelForCausalLM

class LlamaInstructInferer:
    def __init__(self, model_path):
        print(f"[INFO] Loading LLaMA Instruct model: {model_path}")
        self.tokenizer = AutoTokenizer.from_pretrained(model_path)
        self.model = AutoModelForCausalLM.from_pretrained(
            model_path,
            torch_dtype=torch.float16,
            device_map={"": "cuda"},
        )
        self.model.eval()

    # ---------------------------------------------------------
    # TRAININGSPROMPT 1:1 — unverändert
    # ---------------------------------------------------------
    def build_messages(self, question, reference, answer):
        return [
            {
                "role": "system",
                "content": (
                    "You are a grading model. Given a question, a reference answer and a student's "
                    "answer, output only a number between 0.0 and 1.0. No explanation."
                )
            },
            {
                "role": "user",
                "content": (
                    f"Question:\n{question}\n\n"
                    f"Reference answer:\n{reference}\n\n"
                    f"Provided answer (student):\n{answer}\n\n"
                    f"Return only the grade as a number between 0.0 and 1.0."
                )
            }
        ]

    # ---------------------------------------------------------
    # Extraktion NUR des Assistant-Teils (wichtig!)
    # ---------------------------------------------------------
    def extract_assistant_output(self, full_text):
        # LLaMA 3 ChatTemplate uses <|start_header_id|>assistant<|end_header_id|>
        marker = "<|start_header_id|>assistant<|end_header_id|>"
        if marker in full_text:
            return full_text.split(marker)[-1].strip()

        # Fallback: assistant token plain text
        if "assistant" in full_text:
            return full_text.split("assistant")[-1].strip()

        return full_text.strip()

    # ---------------------------------------------------------
    # Reiner Zahl-Parser (KEIN Clamping!)
    # ---------------------------------------------------------
    def extract_number(self, text):
        nums = re.findall(r"[-+]?\d*\.\d+|\d+", text)
        for n in nums:
            try:
                return float(n)
            except:
                continue
        return None  # Modell hat KEINE Zahl geliefert

    # ---------------------------------------------------------
    # INFER — wissenschaftlich korrekte Extraktion
    # ---------------------------------------------------------
    def infer(self, question, reference, answer):
        messages = self.build_messages(question, reference, answer)

        # ChatTemplate 1:1 wie von Meta vorgesehen
        prompt = self.tokenizer.apply_chat_template(
            messages,
            add_generation_prompt=True,
            tokenize=False
        )

        inputs = self.tokenizer(prompt, return_tensors="pt").to("cuda")

        start = time.time()
        with torch.no_grad():
            output = self.model.generate(
                **inputs,
                max_new_tokens=12,
                do_sample=False,
                temperature=1e-7,
                top_p=1.0,
            )
        duration_ms = (time.time() - start) * 1000.0

        # Decode
        full_text = self.tokenizer.decode(output[0], skip_special_tokens=False)

        # Extrahiere NUR den Assistant-Output
        assistant_part = self.extract_assistant_output(full_text)

        # Numerischer Wert extrahieren (kein clamp)
        pred = self.extract_number(assistant_part)

        return assistant_part, pred, duration_ms
