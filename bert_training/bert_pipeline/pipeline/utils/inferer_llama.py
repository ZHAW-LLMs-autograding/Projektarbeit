import time
import torch
from tqdm import tqdm
import os
import json
import re


class InfererLLaMA:
    """
    Universal-Inferer für ALLE LLaMA-Modelle (Base, Instruct, Fine-Tuned),
    mit exakt dem Prompt-Template aus dem Training.
    """

    # -----------------------------------------------------------
    # TRAININGSPROMPT → 1:1 aus Kevin's Training
    # -----------------------------------------------------------
    PROMPT_TEMPLATE = (
        "[INST]\n"
        "Question:\n{q}\n\n"
        "Reference answer:\n{r}\n\n"
        "Provided answer:\n{p}\n"
        "[/INST]\n"
        "Predict the grade based on the given question with the reference answer "
        "and the provided answer made by the student and return a normalized grade between 0.0 and 1.0."
    )

    # -----------------------------------------------------------
    # Constructor
    # -----------------------------------------------------------
    def __init__(self, model, tokenizer=None, device="cuda", run_name="default_run"):
        self.model = model
        self.tokenizer = tokenizer
        self.device = device
        self.run_name = run_name

        # Model vorbereiten
        try:
            self.model.eval()
        except:
            pass

        try:
            self.model.to(self.device)
        except:
            pass

        # Output-Verzeichnis
        self.out_dir = os.path.join("metriken", "benchmarks", run_name)
        os.makedirs(self.out_dir, exist_ok=True)

    # -----------------------------------------------------------
    # Hilfsfunktion: Prompt erzeugen
    # -----------------------------------------------------------
    def build_prompt(self, entry):
        return self.PROMPT_TEMPLATE.format(
            q=entry["question"],
            r=entry["reference_answer"],
            p=entry["provided_answer"]
        )

    # -----------------------------------------------------------
    # Einzelne Prediction
    # -----------------------------------------------------------
    def predict_single(self, prompt):
        start = time.time()

        encoded = self.tokenizer(prompt, return_tensors="pt").to(self.device)

        try:
            output = self.model.generate(
                **encoded,
                max_new_tokens=32,
                do_sample=False,
                top_p=1.0,
                pad_token_id=self.tokenizer.eos_token_id,
                eos_token_id=self.tokenizer.eos_token_id,
            )

            # Nur die neuen Tokens dekodieren
            input_len = encoded["input_ids"].shape[-1]
            gen_ids = output[0][input_len:]
            decoded = self.tokenizer.decode(gen_ids, skip_special_tokens=True).strip()

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
    # Score extrahieren → erste Zahl / fraction / None
    # -----------------------------------------------------------
    def extract_score(self, text):
        if not text:
            return None

        t = text.lower().strip()

        # 1) Fraction: 7/10 → nimm Zähler
        m = re.search(r"\b([-+]?\d*\.?\d+)\s*/\s*\d+", t)
        if m:
            return float(m.group(1))

        # 2) "x out of y"
        m = re.search(r"\b([-+]?\d*\.?\d+)\s+out of\s+\d+", t)
        if m:
            return float(m.group(1))

        # 3) Erste Zahl
        m = re.search(r"[-+]?\d*\.?\d+", t)
        if m:
            return float(m.group(0))

        return None

    # -----------------------------------------------------------
    # Dataset verarbeiten
    # -----------------------------------------------------------
    def predict_dataset(self, dataset, save=True):
        results = []

        for entry in tqdm(dataset, desc="Running Inference"):
            prompt = self.build_prompt(entry)
            out = self.predict_single(prompt)

            results.append({
                "question": entry["question"],
                "reference_answer": entry["reference_answer"],
                "provided_answer": entry["provided_answer"],
                "y_true": float(entry["label"]),
                "y_pred": out["pred"],   # kann float oder None sein
                "source": entry["source"],
                "inference_time_ms": out["inference_time_ms"],
                "raw_output_text": out["raw_output_text"],
            })

        if save:
            path = os.path.join(self.out_dir, "predictions.json")
            with open(path, "w") as f:
                json.dump(results, f, indent=4)
            print(f"[OK] Predictions gespeichert unter: {path}")

        return results

    # -----------------------------------------------------------
    def benchmark_runtime(self, dataset):
        start = time.time()
        _ = self.predict_dataset(dataset, save=False)
        end = time.time()

        data = {"benchmark_time_s": float(end - start)}
        path = os.path.join(self.out_dir, "runtime.json")

        with open(path, "w") as f:
            json.dump(data, f, indent=4)

        print(f"[OK] Runtime gespeichert unter: {path}")
        return data
