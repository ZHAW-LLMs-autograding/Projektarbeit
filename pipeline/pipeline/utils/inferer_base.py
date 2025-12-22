import time
import torch
from tqdm import tqdm
import os
import json
import re


class InfererBase:
    """
    Inferer für LLaMA BASE Modelle (kein Chat-Template!)
    Erwartet reinen Text-Prompt.
    """

    # -----------------------------------------------------------
    # Base Prompt Builder (kein apply_chat_template!)
    # -----------------------------------------------------------
    def build_base_prompt(self, entry):
        """
        Baut einen einfachen Prompt (wie wir manuell getestet haben).
        """
        return (
            f"Question:\n{entry['question']}\n\n"
            f"Reference answer:\n{entry['reference_answer']}\n\n"
            f"Student answer:\n{entry['provided_answer']}\n\n"
            "Predict the grade for the student answer on a normalized scale between 0.0 and 1.0.\n"
            "Return only a single number."
        )

    # -----------------------------------------------------------
    # Constructor
    # -----------------------------------------------------------
    def __init__(self, model, tokenizer, device="cuda", run_name="llama_base"):
        self.model = model
        self.tokenizer = tokenizer
        self.device = device
        self.run_name = run_name

        try:
            self.model.eval()
        except:
            pass

        try:
            self.model.to(self.device)
        except:
            pass

        # Output directory
        self.benchmark_dir = os.path.join("metriken", "benchmarks", self.run_name)
        os.makedirs(self.benchmark_dir, exist_ok=True)

    # -----------------------------------------------------------
    # Move encoded batch to device
    # -----------------------------------------------------------
    def _move_to_device(self, encoded):
        if hasattr(encoded, "to"):
            return encoded.to(self.device)

        if isinstance(encoded, dict):
            for k, v in encoded.items():
                if isinstance(v, torch.Tensor):
                    encoded[k] = v.to(self.device)
        return encoded

    # -----------------------------------------------------------
    # Single prediction
    # -----------------------------------------------------------
    def predict_single(self, prompt_text):
        start = time.time()

        encoded = self.tokenizer(prompt_text, return_tensors="pt")
        encoded = self._move_to_device(encoded)

        # BASE → generate()
        try:
            output = self.model.generate(
                **encoded,
                max_new_tokens=32,
                do_sample=False,
                top_p=1.0,
                pad_token_id=self.tokenizer.eos_token_id,
                eos_token_id=self.tokenizer.eos_token_id,
            )

            # Nur neu generierte Tokens dekodieren
            input_len = encoded["input_ids"].shape[-1]
            generated_ids = output[0][input_len:]
            decoded = self.tokenizer.decode(generated_ids, skip_special_tokens=True).strip()

            pred = self.extract_score(decoded)

        except Exception:
            decoded = ""
            pred = None

        end = time.time()

        return {
            "pred": pred,
            "raw_output_text": decoded,
            "inference_time_ms": (end - start) * 1000.0,
        }

    # -----------------------------------------------------------
    # Predictions für gesamtes Dataset
    # -----------------------------------------------------------
    def predict_dataset(self, dataset, save_predictions=True):
        results = []

        for entry in tqdm(dataset, desc="Running LLaMA Base Inference"):
            prompt = self.build_base_prompt(entry)
            out = self.predict_single(prompt)

            results.append({
                "question": entry["question"],
                "reference_answer": entry["reference_answer"],
                "provided_answer": entry["provided_answer"],
                "y_true": float(entry["label"]),
                "y_pred": out["pred"],
                "source": entry.get("source", "unknown"),
                "raw_output_text": out["raw_output_text"],
                "inference_time_ms": out["inference_time_ms"],
            })

        if save_predictions:
            fpath = os.path.join(self.benchmark_dir, "predictions.json")
            with open(fpath, "w") as f:
                json.dump(results, f, indent=4)
            print(f"[OK] Predictions gespeichert unter {fpath}")

        return results

    # -----------------------------------------------------------
    # Benchmark
    # -----------------------------------------------------------
    def benchmark_runtime(self, dataset):
        start = time.time()
        self.predict_dataset(dataset, save_predictions=False)
        end = time.time()

        runtime_s = float(end - start)
        out_path = os.path.join(self.benchmark_dir, "runtime.json")

        with open(out_path, "w") as f:
            json.dump({"runtime_s": runtime_s}, f, indent=4)

        print(f"[OK] Runtime gespeichert unter {out_path}")
        return runtime_s

    # -----------------------------------------------------------
    # Score extraktion (BASE)
    # -----------------------------------------------------------
    def extract_score(self, text):
        if not text or not isinstance(text, str):
            return None

        t = text.lower().strip()

        # Case: 7/10 → take 7
        m = re.search(r"\b([-+]?\d*\.?\d+)\s*/\s*\d", t)
        if m:
            return float(m.group(1))

        # Case: "x out of y"
        m = re.search(r"\b([-+]?\d*\.?\d+)\s+out of", t)
        if m:
            return float(m.group(1))

        # Case: first standalone number
        m = re.search(r"[-+]?\d*\.?\d+", t)
        if m:
            return float(m.group(0))

        return None
