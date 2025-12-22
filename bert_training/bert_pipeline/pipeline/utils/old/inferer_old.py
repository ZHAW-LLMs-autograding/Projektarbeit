import time
import torch
from tqdm import tqdm
import os
import json
import re


class Inferer:
    """
    Modell-agnostischer Inferer.
    Unterstützt:
    - CausalLM Modelle (generate)
    - BERT / Regression (forward / logits)
    - FakeModel für Tests
    Speichert Ergebnisse in:
        metriken/benchmarks/<run_name>/
    """

    def __init__(self, model, tokenizer=None, device="cuda", run_name="default_run"):
        self.model = model
        self.tokenizer = tokenizer
        self.device = device
        self.run_name = run_name

        # Setzt Model in eval-Mode und auf das richtige Device
        try:
            self.model.eval()
        except:
            pass

        try:
            self.model.to(self.device)
        except:
            pass

        # Output-Verzeichnis erstellen
        self.benchmark_dir = os.path.join("metriken", "benchmarks", self.run_name)
        os.makedirs(self.benchmark_dir, exist_ok=True)

    # -----------------------------------------------------------
    # Helper: encoded Inputs robust auf device verschieben
    # -----------------------------------------------------------
    def _move_to_device(self, encoded):
        # Fall 1: echtes HF BatchEncoding → besitzt .to()
        if hasattr(encoded, "to"):
            return encoded.to(self.device)

        # Fall 2: dict → iteriere über Einträge
        if isinstance(encoded, dict):
            for k, v in encoded.items():
                if isinstance(v, torch.Tensor):
                    encoded[k] = v.to(self.device)
            return encoded

        # Fall 3: FakeBatch mit .to()
        if hasattr(encoded, "to"):
            return encoded.to(self.device)

        # Sonst: unverändert zurück
        return encoded

    # -----------------------------------------------------------
    # 1) Prediction für einzelne Input
    # -----------------------------------------------------------
    def predict_single(self, text_input):
        """
        Führt ein einzelnes Modell-Forward oder generate() aus
        und misst die Inferenzzeit.
        """

        start = time.time()

        # Tokenisieren
        encoded = self.tokenizer(text_input, return_tensors="pt")

        # Token auf Gerät verschieben
        encoded = self._move_to_device(encoded)

        # -----------------------------------------------
        # Fall A: CausalLM (LLaMA, Mixtral, GPT)
        # -----------------------------------------------
        if hasattr(self.model, "generate"):
            try:
                output = self.model.generate(**encoded, max_new_tokens=10)
                decoded = self.tokenizer.decode(output[0], skip_special_tokens=True)
                pred = self.extract_score(decoded)

            except Exception:
                # Fallback für FakeModel: generate() gibt evtl. Tensor / Liste zurück
                try:
                    decoded = str(output)
                    pred = self.extract_score(decoded)
                except:
                    pred = 0.0

        # -----------------------------------------------
        # Fall B: Regression / BERT-like Modelle
        # -----------------------------------------------
        else:
            with torch.no_grad():
                out = self.model(**encoded)
                if hasattr(out, "logits"):
                    pred = float(out.logits.squeeze().item())
                else:
                    pred = 0.0

        end = time.time()
        inference_ms = (end - start) * 1000.0

        return {
            "pred": float(pred),
            "inference_time_ms": inference_ms
        }

    # -----------------------------------------------------------
    # 2) Predictions für gesamtes Dataset
    # -----------------------------------------------------------
    def predict_dataset(self, dataset, save_predictions=True):
        """
        dataset: Liste von dicts:
            { "text": "...", "label": 0.8, "source": "SAF" }
        """

        results = []

        for entry in tqdm(dataset, desc="Running Inference"):
            text = entry["text"]
            y_true = float(entry["label"])
            source = entry.get("source", "unknown")

            out = self.predict_single(text)
            y_pred = out["pred"]

            results.append({
                "y_true": y_true,
                "y_pred": y_pred,
                "source": source
            })

        # Optional: Predictions speichern
        if save_predictions:
            pred_path = os.path.join(self.benchmark_dir, "predictions.json")
            with open(pred_path, "w") as f:
                json.dump(results, f, indent=4)
            print(f"[OK] Predictions gespeichert unter: {pred_path}")

        return results

    # -----------------------------------------------------------
    # 3) Benchmark-Zeit
    # -----------------------------------------------------------
    def benchmark_runtime(self, dataset):
        start = time.time()
        _ = self.predict_dataset(dataset, save_predictions=False)
        end = time.time()

        runtime_s = float(end - start)

        out_path = os.path.join(self.benchmark_dir, "runtime.json")
        with open(out_path, "w") as f:
            json.dump({"benchmark_time_s": runtime_s}, f, indent=4)

        print(f"[OK] Benchmarkzeit gespeichert unter: {out_path}")

        return {"benchmark_time_s": runtime_s}

    # -----------------------------------------------------------
    # 4) Score aus Text extrahieren
    # -----------------------------------------------------------
    def extract_score(self, text):
        """
        Extrahiert die letzte float-Zahl aus dem LLM-Output.
        Beispiele:
            "The score is 0.74"
            "0.87"
            "Grade: 0.45 accuracy"
        """
        matches = re.findall(r"[-+]?\d*\.\d+|\d+", text)
        if not matches:
            return 0.0
        return float(matches[-1])
