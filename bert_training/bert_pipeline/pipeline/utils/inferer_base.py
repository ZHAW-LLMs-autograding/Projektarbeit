import torch
import time
import re
from transformers import AutoTokenizer, AutoModelForCausalLM

class LlamaBaseInferer:
    def __init__(self, model_path):
        print(f"[INFO] Loading Base LLaMA model from: {model_path}")
        self.tokenizer = AutoTokenizer.from_pretrained(model_path)
        self.model = AutoModelForCausalLM.from_pretrained(
            model_path,
            torch_dtype=torch.float16,
            device_map={"": "cuda"},
        )
        self.model.eval()

    def build_prompt(self, question, reference, answer):
        prompt = f"""
[INST]
Question:
{question}

Reference answer:
{reference}

Provided answer:
{answer}
[/INST]

Return only a single number between 0.0 and 1.0.
If you cannot determine a number, output nothing.
""".strip()
        return prompt

    def extract_number(self, text):
        """
        Extracts the FIRST valid float in output,
        but does NOT clamp it to [0,1].
        """
        numbers = re.findall(r"[-+]?\d*\.\d+|\d+", text)

        for n in numbers:
            try:
                val = float(n)
                return val   # even if <0 or >1
            except:
                continue

        return None  # no valid number

    def infer(self, question, reference, answer):
        prompt = self.build_prompt(question, reference, answer)

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
        text = self.tokenizer.decode(output[0], skip_special_tokens=True)

        pred = self.extract_number(text)

        return text, pred, duration_ms
