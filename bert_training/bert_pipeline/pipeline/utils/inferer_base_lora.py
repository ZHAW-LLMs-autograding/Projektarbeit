import torch
import time
import re
from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import PeftModel

class LlamaBaseLoraInferer:
    """
    Generative LoRA-Inferenz für Base-LLaMA (kein reg_head).
    """

    def __init__(self, base_model_path, lora_path):
        print("[INFO] Loading base model:", base_model_path)
        print("[INFO] Loading LoRA adapter:", lora_path)

        self.tokenizer = AutoTokenizer.from_pretrained(base_model_path)

        base = AutoModelForCausalLM.from_pretrained(
            base_model_path,
            torch_dtype=torch.float16,
            device_map={"": "cuda"},
        )

        self.model = PeftModel.from_pretrained(
            base,
            lora_path,
            torch_dtype=torch.float16,
        ).eval()

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

Return only the grade as a number between 0.0 and 1.0.
""".strip()

    def extract_number(self, text):
        nums = re.findall(r"[-+]?\d*\.\d+|\d+", text)
        return float(nums[0]) if nums else None

    def infer(self, question, reference, answer):
        prompt = self.build_prompt(question, reference, answer)
        inputs = self.tokenizer(prompt, return_tensors="pt").to("cuda")

        start = time.time()

        output_ids = self.model.generate(
            **inputs,
            max_new_tokens=8,
            do_sample=False,
            temperature=1e-7,
            top_p=1.0,
            eos_token_id=self.tokenizer.eos_token_id,
        )

        duration_ms = (time.time() - start) * 1000.0

        text = self.tokenizer.decode(output_ids[0], skip_special_tokens=True)
        pred = self.extract_number(text)

        return text, pred, duration_ms
