from __future__ import annotations

from pathlib import Path
import torch


def load_checkpoint(path: Path):
    if not path.exists():
        raise FileNotFoundError(f"Checkpoint nicht gefunden: {path}")
    print(f"[INFO] Lade Checkpoint: {path}")
    ckpt = torch.load(path, map_location="cpu")
    model_state = ckpt.get("model", None)
    reg_head_state = ckpt.get("reg_head", None)
    epoch = ckpt.get("epoch", None)
    return {
        "path": path,
        "epoch": epoch,
        "model": model_state,
        "reg_head": reg_head_state,
    }


def tensor_diff_stats(t1: torch.Tensor, t2: torch.Tensor):
    """Gibt Normen und einfache Statistiken für die Differenz zurück."""
    diff = t2 - t1
    return {
        "norm_1": float(t1.norm().item()),
        "norm_2": float(t2.norm().item()),
        "norm_diff": float(diff.norm().item()),
        "mean_abs_diff": float(diff.abs().mean().item()),
        "max_abs_diff": float(diff.abs().max().item()),
    }


def compare_reg_head(ckpt2, ckpt18):
    print("\n=== 🔎 Vergleich: Regressionskopf (reg_head) ===")

    h2 = ckpt2["reg_head"]
    h18 = ckpt18["reg_head"]

    if h2 is None or h18 is None:
        print("⚠️ In mindestens einem Checkpoint existiert kein 'reg_head'.")
        return

    keys2 = set(h2.keys())
    keys18 = set(h18.keys())
    if keys2 != keys18:
        print("⚠️ reg_head-Keys unterscheiden sich zwischen den Checkpoints!")
        print("  Nur in ckpt2 :", keys2 - keys18)
        print("  Nur in ckpt18:", keys18 - keys2)

    for k in sorted(keys2 & keys18):
        t2 = h2[k].detach().float()
        t18 = h18[k].detach().float()
        stats = tensor_diff_stats(t2, t18)
        print(f"\n  Param: {k}")
        print(f"    ||ckpt2||      = {stats['norm_1']:.6f}")
        print(f"    ||ckpt18||     = {stats['norm_2']:.6f}")
        print(f"    ||diff||       = {stats['norm_diff']:.6f}")
        print(f"    mean|diff|     = {stats['mean_abs_diff']:.6e}")
        print(f"    max|diff|      = {stats['max_abs_diff']:.6e}")


def compare_lora_params(ckpt2, ckpt18):
    print("\n=== 🔎 Vergleich: LoRA-Parameter im Modell-State ===")

    m2 = ckpt2["model"]
    m18 = ckpt18["model"]

    if m2 is None or m18 is None:
        print("⚠️ In mindestens einem Checkpoint existiert kein 'model'-State.")
        return

    keys2 = set(m2.keys())
    keys18 = set(m18.keys())
    shared_keys = keys2 & keys18

    def is_lora_key(k: str) -> bool:
        kl = k.lower()
        return ("lora_" in kl) or ("lora_a" in kl) or ("lora_b" in kl)

    lora_keys = [k for k in shared_keys if is_lora_key(k)]

    if not lora_keys:
        print("⚠️ Keine LoRA-Keys gefunden (Keys ohne 'lora_' im Namen?).")
        return

    print(f"[INFO] Anzahl gemeinsamer LoRA-Parameter: {len(lora_keys)}")

    total_params = 0
    changed_params = 0
    sum_norm_diff = 0.0
    sum_mean_abs_diff = 0.0
    max_abs_diff_global = 0.0

    # Optional: ein paar Beispiel-Parameter detailliert ausgeben
    example_keys = lora_keys[:5]

    for k in lora_keys:
        t2 = m2[k].detach().float()
        t18 = m18[k].detach().float()
        if t2.shape != t18.shape:
            print(f"⚠️ Shape-Mismatch bei {k}: {t2.shape} vs {t18.shape}")
            continue

        stats = tensor_diff_stats(t2, t18)

        n_params = t2.numel()
        total_params += n_params

        if stats["norm_diff"] > 0:
            changed_params += n_params

        sum_norm_diff += stats["norm_diff"]
        sum_mean_abs_diff += stats["mean_abs_diff"]
        max_abs_diff_global = max(max_abs_diff_global, stats["max_abs_diff"])

        if k in example_keys:
            print(f"\n  Beispiel-Param: {k}")
            print(f"    Shape          = {tuple(t2.shape)}")
            print(f"    ||ckpt2||      = {stats['norm_1']:.6f}")
            print(f"    ||ckpt18||     = {stats['norm_2']:.6f}")
            print(f"    ||diff||       = {stats['norm_diff']:.6f}")
            print(f"    mean|diff|     = {stats['mean_abs_diff']:.6e}")
            print(f"    max|diff|      = {stats['max_abs_diff']:.6e}")

    if total_params == 0:
        print("⚠️ Konnte keine LoRA-Parameter zählen.")
        return

    frac_changed = changed_params / total_params

    print("\n--- Zusammenfassung LoRA-Parameter ---")
    print(f"  Gesamtanzahl Parameter (LoRA): {total_params}")
    print(f"  Anzahl Parameter mit Veränderung: {changed_params}")
    print(f"  Anteil veränderter Parameter: {frac_changed*100:.2f}%")
    print(f"  Summe ||diff|| über alle LoRA-Parameter: {sum_norm_diff:.6f}")
    print(f"  Mittelwert mean|diff| pro LoRA-Param (über Keys gemittelt): {sum_mean_abs_diff/len(lora_keys):.6e}")
    print(f"  Globales max|diff|: {max_abs_diff_global:.6e}")


def main():
    # Dieses Script liegt in: projektarbeit/pipeline/
    here = Path(__file__).resolve().parent      # .../projektarbeit/pipeline
    project_root = here.parent                  # .../projektarbeit

    ckpt_dir = project_root / "checkpoints"
    ckpt2_path = ckpt_dir / "model_epoch2.pt"
    ckpt18_path = ckpt_dir / "model_epoch18.pt"

    print(f"[INFO] Checkpoint-Verzeichnis: {ckpt_dir}")
    ckpt2 = load_checkpoint(ckpt2_path)
    ckpt18 = load_checkpoint(ckpt18_path)

    print(f"\n[INFO] Epoch in ckpt2 : {ckpt2['epoch']}")
    print(f"[INFO] Epoch in ckpt18: {ckpt18['epoch']}")

    compare_reg_head(ckpt2, ckpt18)
    compare_lora_params(ckpt2, ckpt18)


if __name__ == "__main__":
    main()
