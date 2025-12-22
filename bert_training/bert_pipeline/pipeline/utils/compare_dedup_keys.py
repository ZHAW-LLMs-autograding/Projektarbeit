import json

FILE = "/home/sercan/asag_pipeline/pipeline/metriken/benchmarks/vectormodels/char_wb_3_5_min2/predictions_repaired_withoutDuplicates.json"


def load_predictions(path):
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)
    if isinstance(data, dict) and "predictions" in data:
        return data["predictions"]
    return data


# ---------------------------------------------------------
# 1) Dein (falscher) Key – mit ALLEN Feldern außer inference_time_ms
# ---------------------------------------------------------
def make_big_key(entry):
    return (
        entry.get("question"),
        entry.get("reference_answer"),
        entry.get("provided_answer"),
        entry.get("y_true"),
        entry.get("y_pred"),
        entry.get("source"),
        entry.get("raw_output_text"),
    )


# ---------------------------------------------------------
# 2) Der richtige ASAG-Key – nur Datensatzfelder
# ---------------------------------------------------------
def make_small_key(entry):
    return (
        entry.get("question"),
        entry.get("reference_answer"),
        entry.get("provided_answer"),
        entry.get("y_true"),
    )


def find_duplicates(entries, key_fn):
    seen = set()
    dups = []

    for idx, entry in enumerate(entries):
        key = key_fn(entry)
        if key in seen:
            dups.append((idx, entry))
        else:
            seen.add(key)

    return dups


if __name__ == "__main__":
    data = load_predictions(FILE)

    print(f"📌 Gesamt Einträge: {len(data)}")

    # Finde Duplikate mit beiden Keys
    big_dups = find_duplicates(data, make_big_key)
    small_dups = find_duplicates(data, make_small_key)

    print("\n=============================")
    print(" DUPLIKAT-ANZAHLEN")
    print("=============================")
    print(f"🔵 Großes Key (deiner Logik): {len(big_dups)}")
    print(f"🟢 Kleines Key (korrekt):     {len(small_dups)}")

    # Welche nur im großen Key vorkommen?
    big_keys = {make_big_key(e) for (_, e) in big_dups}
    small_keys = {make_small_key(e) for (_, e) in small_dups}

    print("\n=============================")
    print(" VERGLEICH")
    print("=============================")

    # künstliche Duplikate = in groß, aber NICHT in klein
    artificial = []
    for idx, e in big_dups:
        if make_small_key(e) not in small_keys:
            artificial.append((idx, e))

    print(f"❌ Künstliche Duplikate (nur großer Key): {len(artificial)}")

    # verpasste Duplikate = in klein, aber NICHT in groß
    missed = []
    big_small_keys = {make_small_key(e) for (_, e) in big_dups}
    for idx, e in small_dups:
        if make_small_key(e) not in big_small_keys:
            missed.append((idx, e))

    print(f"⚠️ Verpasste echte Duplikate (nur kleiner Key): {len(missed)}")

    print("\n=============================")
    print(" Beispiele (max 5)")
    print("=============================")

    print("\n❌ Künstliche Duplikate (nur großer Key):")
    for idx, e in artificial[:5]:
        print(f"- Index {idx} | Frage: {str(e['question'])[:50]}")

    print("\n⚠️ Verpasste echte Duplikate (nur kleiner Key):")
    for idx, e in missed[:5]:
        print(f"- Index {idx} | Frage: {str(e['question'])[:50]}")

    print("\n🟢 Gemeinsame Duplikate (beide Keys finden sie):")
    common = []
    small_dups_set = {make_small_key(e) for (_, e) in small_dups}
    for idx, e in big_dups:
        if make_small_key(e) in small_dups_set:
            common.append((idx, e))

    for idx, e in common[:5]:
        print(f"- Index {idx} | Frage: {str(e['question'])[:50]}")
