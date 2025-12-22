import json


def load_predictions(path):
    """
    Lädt Predictions im Format:
    - { "predictions": [ ... ] }
    - [ ... ]
    und gibt IMMER eine Liste zurück.
    """
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)

    # Fall 1: Datei ist ein Dict mit "predictions"
    if isinstance(data, dict) and "predictions" in data:
        return data["predictions"]

    # Fall 2: Datei ist eine Liste
    if isinstance(data, list):
        return data

    raise ValueError(f"Unbekanntes JSON-Format in {path}")


def save_predictions(path, predictions):
    """
    Speichert Predictions als Liste (wie BERT-Format).
    """
    with open(path, "w", encoding="utf-8") as f:
        json.dump(predictions, f, ensure_ascii=False, indent=2)


def make_key(entry):
    """4-Feld-Matching (robust)."""
    return (
        (entry.get("question") or "").strip(),
        (entry.get("reference_answer") or "").strip(),
        (entry.get("provided_answer") or "").strip(),
        str(entry.get("y_true")),
    )


def repair_sources(broken_path, reference_paths, output_path):
    print(f"🔍 Lade BERT-Datei: {broken_path}")
    broken = load_predictions(broken_path)

    print("🔍 Lade Referenzdateien...")
    reference_map = {}

    for ref_path in reference_paths:
        print(f"   → {ref_path}")
        ref_preds = load_predictions(ref_path)

        for e in ref_preds:
            key = make_key(e)
            reference_map[key] = e.get("source")

    print(f"📌 Gesamt {len(reference_map)} Referenzeinträge geladen.")

    repaired = []
    unknown_total = 0
    repaired_count = 0

    for entry in broken:
        if entry.get("source") != "unknown":
            repaired.append(entry)
            continue

        unknown_total += 1
        key = make_key(entry)

        if key in reference_map:
            new_entry = dict(entry)
            new_entry["source"] = reference_map[key]
            repaired.append(new_entry)
            repaired_count += 1
        else:
            print("⚠️ Kein Match gefunden für:")
            print(f"   Frage: {entry.get('question')[:80]}...")
            repaired.append(entry)

    print("\n📊 Ergebnis:")
    print(f"  Unknown vorher : {unknown_total}")
    print(f"  Davon repariert: {repaired_count}")
    print(f"  Fehlende Matches: {unknown_total - repaired_count}")

    print(f"\n💾 Speichere repariertes File nach:\n{output_path}")
    save_predictions(output_path, repaired)


if __name__ == "__main__":
    repair_sources(
        broken_path="/home/sercan/asag_pipeline/pipeline/metriken/benchmarks/vectormodels/char_wb_3_5_min2/predictions.json",
        reference_paths=[
            "/home/sercan/asag_pipeline/pipeline/metriken/benchmarks/llama_base_run/predictions.json",
            "/home/sercan/asag_pipeline/pipeline/metriken/benchmarks/llama_instruct_run/predictions.json",
        ],
        output_path="/home/sercan/asag_pipeline/pipeline/metriken/benchmarks/vectormodels/char_wb_3_5_min2/predictions_repaired.json"
    )
