import time
import torch
from tqdm import tqdm
import os
import json
import re


class Inferer:
    """
    Modell-agnostischer Inferer für ASAG.
    Jetzt kompatibel mit:
    - Llama 3.1 Instruct (Chat-Prompts)
    - beliebigen CausalLM-Modellen
    - Regression/BERT-ähnlichen Modellen

    Speichert vollständige Rohdaten unter:
        metriken/benchmarks/<run_name>/
    """

    # -----------------------------------------------------------
    # 0) Llama-Instruct Prompt Builder
    # -----------------------------------------------------------
    def build_instruct_prompt(self, entry):
        system_msg = (
            "You are a grading model. Given a question, a reference answer and a student's "
            "answer, output only a single number between 0.0 and 1.0. No explanation."
        )

        messages = [
            {"role": "system", "content": system_msg},
            {
                "role": "user",
                "content":
                (
                    f"Question:\n{entry['question']}\n\n"
                    f"Reference answer:\n{entry['reference_answer']}\n\n"
                    f"Provided answer (student):\n{entry['provided_answer']}\n\n"
                    "Return only the grade as a number between 0.0 and 1.0."
                )
            }
        ]

        return self.tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
        )

    # -----------------------------------------------------------
    # Constructor
    # -----------------------------------------------------------
    def __init__(self, model, tokenizer=None, device="cuda", run_name="default_run"):
        self.model = model
        self.tokenizer = tokenizer
        self.device = device
        self.run_name = run_name

        # eval mode
        try:
            self.model.eval()
        except:
            pass

        try:
            self.model.to(self.device)
        except:
            pass

        # output dir
        self.benchmark_dir = os.path.join("metriken", "benchmarks", self.run_name)
        os.makedirs(self.benchmark_dir, exist_ok=True)

    # -----------------------------------------------------------
    # Helper: safely move encoded tensors to device
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
    # 1) Prediction für ein einzelnes Sample
    # -----------------------------------------------------------
    def predict_single(self, text_input):
        start = time.time()

        encoded = self.tokenizer(text_input, return_tensors="pt")
        encoded = self._move_to_device(encoded)

        # -------------------------------------------
        # CausalLM (Llama / GPT / Mixtral)
        # -------------------------------------------
        if hasattr(self.model, "generate"):
            try:
                output = self.model.generate(
                    **encoded,
                    max_new_tokens=32,
                    do_sample=False,
                    top_p=1.0,
                    pad_token_id=self.tokenizer.pad_token_id,
                    eos_token_id=self.tokenizer.eos_token_id,
                )

                # NUR neu generierte Tokens dekodieren
                input_len = encoded["input_ids"].shape[-1]
                generated_ids = output[0][input_len:]
                decoded = self.tokenizer.decode(
                    generated_ids, skip_special_tokens=True
                ).strip()

                pred = self.extract_score(decoded)

            except Exception:
                try:
                    decoded = str(output)
                    pred = self.extract_score(decoded)
                except:
                    decoded = ""
                    pred = None

        # -------------------------------------------
        # Regression / BERT-artige Modelle
        # -------------------------------------------
        else:
            with torch.no_grad():
                out = self.model(**encoded)
                if hasattr(out, "logits"):
                    pred = float(out.logits.squeeze().item())
                    decoded = str(pred)
                else:
                    pred = None
                    decoded = ""

        end = time.time()

        return {
            "pred": pred if pred is not None else None,
            "raw_output_text": decoded,
            "inference_time_ms": (end - start) * 1000.0
        }

    # -----------------------------------------------------------
    # 2) Predictions über gesamtes Dataset
    # -----------------------------------------------------------
    def predict_dataset(self, dataset, save_predictions=True):
        results = []

        for entry in tqdm(dataset, desc="Running Inference"):
            text = self.build_instruct_prompt(entry)

            y_true = float(entry["label"])
            source = entry.get("source", "unknown")

            out = self.predict_single(text)
            y_pred = out["pred"]

            results.append({
                "y_true": y_true,
                "y_pred": y_pred if y_pred is not None else None,
                "source": source,
                "question": entry["question"],
                "reference_answer": entry["reference_answer"],
                "provided_answer": entry["provided_answer"],
                "raw_output_text": out.get("raw_output_text"),
                "inference_time_ms": out.get("inference_time_ms"),
            })

        if save_predictions:
            pred_path = os.path.join(self.benchmark_dir, "predictions.json")
            with open(pred_path, "w") as f:
                json.dump(results, f, indent=4)
            print(f"[OK] Predictions gespeichert unter: {pred_path}")

        return results

    # -----------------------------------------------------------
    # 3) Benchmark Runtime
    # -----------------------------------------------------------
    def benchmark_runtime(self, dataset):
        start = time.time()
        _ = self.predict_dataset(dataset, save_predictions=False)
        end = time.time()

        seconds = float(end - start)

        out_path = os.path.join(self.benchmark_dir, "runtime.json")
        with open(out_path, "w") as f:
            json.dump({"benchmark_time_s": seconds}, f, indent=4)

        print(f"[OK] Benchmarkzeit gespeichert unter: {out_path}")

        return {"benchmark_time_s": seconds}

    # -----------------------------------------------------------
    # 4) Robuste Score-Extraktion
    # -----------------------------------------------------------
    def extract_score(self, text):
        """
        Extrahiert die erste sinnvolle Zahl aus dem generierten Modell-Output.
        Rückgabe:
            float  → gültiger Wert
            None   → wenn keine Zahl gefunden wurde
        """

        if not text or not isinstance(text, str):
            return None

        t = text.lower().strip()

        # 1) Fraction (7/10) → nimm Numerator
        m = re.search(r"\b([-+]?\d*\.?\d+)\s*/\s*([-+]?\d*\.?\d+)\b", t)
        if m:
            return float(m.group(1))

        # 2) "x out of y" → nimm x
        m = re.search(r"\b([-+]?\d*\.?\d+)\s+out of\s+([-+]?\d*\.?\d+)\b", t)
        if m:
            return float(m.group(1))

        # 3) Erste einzelne Zahl
        m = re.search(r"[-+]?\d*\.?\d+", t)
        if m:
            return float(m.group(0))

        # 4) Keine Zahl gefunden
        return None
