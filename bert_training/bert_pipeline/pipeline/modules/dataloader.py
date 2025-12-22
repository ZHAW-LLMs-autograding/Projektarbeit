from __future__ import annotations
import os
import json
import csv
import random
from pathlib import Path
from typing import Any, Dict, List, Optional, Callable

try:
    import torch
    from torch.utils.data import Dataset, DataLoader
except ImportError:
    torch = None
    class Dataset:  # type: ignore
        pass
    class DataLoader:  # type: ignore
        def __init__(self, *args, **kwargs):
            raise ImportError("PyTorch not installed. Install torch to use DataLoader.")

from pipeline.config import load_global_config, effective_data_loader_config


# -----------------------------
# IO helpers
# -----------------------------
def _read_any(path: str) -> List[Dict[str, Any]]:
    ext = os.path.splitext(path)[1].lower()
    if ext in (".jsonl", ".ndjson"):
        with open(path, "r", encoding="utf-8") as f:
            return [json.loads(line) for line in f if line.strip()]
    if ext == ".json":
        with open(path, "r", encoding="utf-8") as f:
            obj = json.load(f)
            return obj["data"] if isinstance(obj, dict) and "data" in obj else obj
    if ext == ".csv":
        with open(path, "r", encoding="utf-8") as f:
            return list(csv.DictReader(f))
    if ext == ".parquet":
        import pandas as pd
        df = pd.read_parquet(path)
        return df.to_dict(orient="records")
    raise ValueError(f"Unsupported file format: {ext}")


# -----------------------------
# Augmentation helpers
# -----------------------------
def _noisify_text(text: str, drop_p: float, swap_p: float, ins_p: float) -> str:
    if not isinstance(text, str) or not text:
        return text
    chars = list(text)
    out, i = [], 0
    while i < len(chars):
        c = chars[i]
        if random.random() < drop_p:
            i += 1
            continue
        if i + 1 < len(chars) and random.random() < swap_p:
            out.extend([chars[i + 1], c])
            i += 2
            continue
        out.append(c)
        i += 1
        if random.random() < ins_p:
            out.append(random.choice(" abcdefghijklmnopqrstuvwxyz"))
    return "".join(out)


def _negate_answer(sample: Dict[str, Any], field: str) -> None:
    ans = sample.get(field)
    if isinstance(ans, bool):
        sample[field] = not ans
    elif isinstance(ans, (int, float)):
        sample[field] = -ans
    elif isinstance(ans, str):
        low = ans.strip().lower()
        if low in {"yes", "true", "y"}:
            sample[field] = "no"
        elif low in {"no", "false", "n"}:
            sample[field] = "yes"
        else:
            sample[field] = f"NOT {ans}"


# -----------------------------
# Dataset
# -----------------------------
class TextQADataset(Dataset):
    """
    Schema-gesteuertes Dataset mit (optional) aktivierten Augmentationen.
    Erwartete Normalform: id, question, context, answer, provided_answer, meta, raw
    """

    def __init__(
        self,
        data: List[Dict[str, Any]],
        split: str,
        schema,
        augment,
        tokenizer: Optional[Callable[[Dict[str, Any]], Dict[str, Any]]] = None,
        seed: int = 42,
    ):
        self.data = data
        self.split = split
        self.schema = schema
        self.augment = augment
        self.tokenizer = tokenizer
        self.rnd = random.Random(seed)

        # globaler Schalter + per-split Schalter
        self.apply_aug = (
            getattr(self.augment, "enabled", True) and (
                (split == "train" and self.augment.apply_on_train)
                or (split == "val" and self.augment.apply_on_val)
                or (split == "test" and self.augment.apply_on_test)
            )
        )

    def __len__(self) -> int:
        return len(self.data)

    def _apply_augmentations(self, row: Dict[str, Any]) -> Dict[str, Any]:
        if not self.apply_aug:
            return row
        r = dict(row)
        # Text-Noise auf Frage/Kontext
        if self.rnd.random() < self.augment.p_text_noise:
            if getattr(self.augment, "noise_on_question", True) and self.schema.question_field in r:
                r[self.schema.question_field] = _noisify_text(
                    r[self.schema.question_field],
                    self.augment.noise_drop_prob,
                    self.augment.noise_swap_prob,
                    self.augment.noise_insert_prob,
                )
            if getattr(self.augment, "noise_on_context", False) and getattr(self.schema, "context_field", None):
                cf = self.schema.context_field
                if cf and cf in r and isinstance(r[cf], str):
                    r[cf] = _noisify_text(
                        r[cf],
                        self.augment.noise_drop_prob,
                        self.augment.noise_swap_prob,
                        self.augment.noise_insert_prob,
                    )
        # Answer negation
        if self.rnd.random() < self.augment.p_negate_answer:
            _negate_answer(r, self.schema.answer_field)
        # Choice shuffle (falls vorhanden)
        if self.rnd.random() < getattr(self.augment, "p_shuffle_choices", 0.0):
            cf = getattr(self.schema, "choices_field", None)
            if cf and isinstance(r.get(cf), list) and len(r[cf]) > 1:
                self.rnd.shuffle(r[cf])
        return r

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        row = self._apply_augmentations(dict(self.data[idx]))

        # Quelle für die Referenzantwort gemäß Schema (z. B. "reference_answer")
        ref = row.get(self.schema.answer_field, "")

        sample = {
            "id": row.get(self.schema.id_field, idx),
            "question": row.get(self.schema.question_field, ""),
            "context": row.get(getattr(self.schema, "context_field", None), ""),
            # ↓↓↓ Kompatibel für beide Welten:
            "answer": ref,                     # bestehender Key (bisher im Batch)
            "reference_answer": ref,           # neuer, expliziter Key für den Trainer
            "provided_answer": row.get(getattr(self.schema, "provided_answer_field", None), ""),
            "meta": {k: row.get(k) for k in self.schema.passthrough_fields if k in row},
            "raw": row,
        }

        # Labels & Weights explizit auf Top-Level heben, falls vorhanden
        if "normalized_grade" in row:
            sample["normalized_grade"] = row["normalized_grade"]
        if getattr(self.schema, "weight_field", None):
            wf = self.schema.weight_field
            if wf and wf in row:
                sample["weight"] = row[wf]
        elif "weight" in row:
            sample["weight"] = row["weight"]

        if self.tokenizer:
            sample = self.tokenizer(sample)
        return sample

# -----------------------------
# DataModule (Controller API)
# -----------------------------
class QADataModule:
    """
    - Lädt Defaults + globale Overrides (inkl. augmentation.enabled)
    - Bietet einfache Controller-Hooks für custom Augmentierungen
    - Liefert PyTorch DataLoader für Trainer
    """

    def __init__(
        self,
        global_config_path: Path,
        tokenizer: Optional[Callable[[Dict[str, Any]], Dict[str, Any]]] = None,
        seed: int = 42,
    ):
        self.global_cfg = load_global_config(global_config_path)
        self.cfg = effective_data_loader_config(self.global_cfg)

        self.schema = self.cfg.schema
        self.augment = self.cfg.augmentation
        self.loader_cfg = self.cfg.loader

        self.data_root = self.global_cfg.paths.data_root
        self.splits = self.global_cfg.dataset.splits  # {"train": "...", "validation": "...", "test": "..."}

        self.tokenizer = tokenizer
        self.seed = seed

        self._cache: Dict[str, List[Dict[str, Any]]] = {}

    # ---------- Loading ----------
    def load_split(self, split: str) -> List[Dict[str, Any]]:
        if split in self._cache:
            return self._cache[split]
        fname = self.splits[split]
        path = str(self.data_root / fname)
        data = _read_any(path)
        self._cache[split] = data
        print(f"[DataLoader] Loaded {split} ({len(data)} rows) from {path}")
        return data

    # ---------- Controller Augmentation Hook ----------
    def augment_split(self, split: str, fn: Callable[[Dict[str, Any]], Dict[str, Any]]) -> None:
        if split not in self._cache:
            self.load_split(split)
        self._cache[split] = [fn(d) for d in self._cache[split]]
        print(f"[DataLoader] Applied custom augmentation to '{split}'.")

    # ---------- Datasets / Loaders ----------
    def dataset(self, split: str) -> TextQADataset:
        data = self.load_split(split)
        canon_split = "val" if split == "validation" else split
        return TextQADataset(
            data=data,
            split=canon_split,
            schema=self.schema,
            augment=self.augment,
            tokenizer=self.tokenizer,
            seed=self.seed,
        )

    def _collate_fn(self, batch: List[Dict[str, Any]]) -> Dict[str, Any]:
        if not batch:
            return {}
        keys = set().union(*(b.keys() for b in batch))
        out: Dict[str, Any] = {}
        for k in keys:
            vals = [b.get(k) for b in batch]
            if torch is not None and all(hasattr(v, "shape") for v in vals if v is not None):
                try:
                    out[k] = torch.stack(vals)  # type: ignore
                    continue
                except Exception:
                    pass
            out[k] = vals
        return out

    def dataloader(self, split: str) -> DataLoader:
        if torch is None:
            raise ImportError("PyTorch not installed.")
        ds = self.dataset(split)
        shuffle = self.loader_cfg.shuffle_train if split in ("train",) else False
        return DataLoader(
            ds,
            batch_size=self.loader_cfg.batch_size,
            shuffle=shuffle,
            num_workers=self.loader_cfg.num_workers,
            pin_memory=self.loader_cfg.pin_memory,
            drop_last=self.loader_cfg.drop_last,
            collate_fn=self._collate_fn,
        )

    # Convenience
    def train_loader(self) -> DataLoader: return self.dataloader("train")
    def val_loader(self) -> DataLoader: return self.dataloader("validation")
    def test_loader(self) -> DataLoader: return self.dataloader("test")

    # ---------- Runtime toggles ----------
    def set_augmentation(self, **kwargs) -> None:
        for k, v in kwargs.items():
            if hasattr(self.augment, k):
                setattr(self.augment, k, v)
            else:
                raise AttributeError(f"Unknown augment param: {k}")
        print(f"[DataLoader] Updated augmentation: {kwargs}")

    def enable_augmentation(self, enabled: bool) -> None:
        if hasattr(self.augment, "enabled"):
            setattr(self.augment, "enabled", enabled)
        else:
            # falls dein Pydantic-Objekt kein 'enabled' hat, setze dynamisch
            self.augment.enabled = enabled  # type: ignore
        print(f"[DataLoader] Augmentation enabled = {enabled}")

    def reload(self) -> None:
        self._cache.clear()
        print("[DataLoader] Cache cleared.")
