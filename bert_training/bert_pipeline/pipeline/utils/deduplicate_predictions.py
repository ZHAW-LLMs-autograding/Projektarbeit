import json

INPUT = "/home/sercan/asag_pipeline/pipeline/metriken/benchmarks/vectormodels/char_wb_3_5_min2/predictions_repaired.json"
OUTPUT = "/home/sercan/asag_pipeline/pipeline/metriken/benchmarks/vectormodels/char_wb_3_5_min2/predictions_repaired_withoutDuplicates.json"


def load_predictions(path):
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)
    # Falls Datei {"predictions": [...]} ist
    if isinstance(data, dict) and "predictions" in data:
        return data["predictions"]
    return data


def save_predictions(path, predictions):
    with open(path, "w", encoding="utf-8") as f:
        json.dump(predictions, f, ensure_ascii=False, indent=2)


def make_key(entry):
    """
    Matching über ALLE Felder außer inference_time_ms.
    """
    return (
        entry.get("question"),
        entry.get("reference_answer"),
        entry.get("provided_answer"),
        entry.get("y_true"),
        entry.get("y_pred"),
        entry.get("source"),
        entry.get("raw_output_text"),
    )


def deduplicate(entries):
    seen = set()
    unique = []

    for e in entries:
        key = make_key(e)
        if key not in seen:
            seen.add(key)
            unique.append(e)

    return unique, len(entries) - len(unique)


if __name__ == "__main__":
    print(f"🔍 Lade Datei: {INPUT}")
    data = load_predictions(INPUT)

    print(f"➡️   Gesamt Einträge vorher: {len(data)}")

    unique, removed = deduplicate(data)

    print(f"🧹   Entfernte Duplikate: {removed}")
    print(f"✅   Einträge nach Dedup: {len(unique)}")

    print(f"💾   Speichere nach: {OUTPUT}")
    save_predictions(OUTPUT, unique)

    print("\n🎉 Fertig! Duplikate erfolgreich entfernt.")
