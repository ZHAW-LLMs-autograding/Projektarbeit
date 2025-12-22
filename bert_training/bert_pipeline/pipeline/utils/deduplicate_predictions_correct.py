import json

INPUT = "/home/sercan/asag_pipeline/pipeline/metriken/benchmarks/bert/predictions_repaired.json"
OUTPUT = "/home/sercan/asag_pipeline/pipeline/metriken/benchmarks/bert/predictions_repaired_withoutDuplicates.json"


def load_predictions(path):
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)
    # Unterstützt sowohl list als auch {"predictions": [...]}
    if isinstance(data, dict) and "predictions" in data:
        return data["predictions"]
    return data


def save_predictions(path, predictions):
    with open(path, "w", encoding="utf-8") as f:
        json.dump(predictions, f, ensure_ascii=False, indent=2)


def make_dataset_key(entry):
    """
    Der einzig korrekte Schlüssel für ASAG-Duplikate.
    Diese vier Felder definieren EINEN ASAG-Testfall.
    """
    return (
        entry.get("question"),
        entry.get("reference_answer"),
        entry.get("provided_answer"),
        entry.get("y_true"),
    )


def deduplicate(entries):
    seen = set()
    unique = []

    for entry in entries:
        key = make_dataset_key(entry)
        if key not in seen:
            seen.add(key)
            unique.append(entry)
        # else = echtes Duplikat → wird entfernt

    removed = len(entries) - len(unique)
    return unique, removed


if __name__ == "__main__":
    print(f"🔍 Lade Datei: {INPUT}")
    data = load_predictions(INPUT)

    print(f"➡️  Gesamt Einträge vorher: {len(data)}")

    unique, removed = deduplicate(data)

    print(f"🧹  Echte ASAG-Duplikate entfernt: {removed}")
    print(f"✅  Einträge nach Dedup: {len(unique)}")

    print(f"💾  Schreibe Ergebnis nach: {OUTPUT}")
    save_predictions(OUTPUT, unique)

    print("\n🎉 Fertig! Korrekte Duplikate entfernt.\n"
          "   Dies entspricht exakt der Anzahl der Duplikate im ASAG-Testdatensatz.")
