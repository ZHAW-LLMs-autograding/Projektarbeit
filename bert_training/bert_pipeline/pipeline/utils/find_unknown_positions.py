import json

FILE = "/home/sercan/asag_pipeline/pipeline/metriken/benchmarks/vectormodels/char_wb_3_5_min2/predictions_repaired.json"

def load_predictions(path):
    """
    Robust loader:
    - { "predictions": [ ... ] }
    - [ ... ]
    - JSONL
    """
    with open(path, "r", encoding="utf-8") as f:
        text = f.read()

    # Standard JSON
    try:
        obj = json.loads(text)
        if isinstance(obj, dict) and "predictions" in obj:
            return obj["predictions"]
        if isinstance(obj, list):
            # List of JSON strings?
            if all(isinstance(x, str) for x in obj):
                return [json.loads(x) for x in obj]
            return obj
    except json.JSONDecodeError:
        pass

    # JSONL fallback
    predictions = []
    for line in text.splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            predictions.append(json.loads(line))
        except:
            pass

    return predictions


def count_unknown_words(entries):
    total = 0
    detailed_hits = []

    for idx, entry in enumerate(entries):
        if not isinstance(entry, dict):
            continue
        for key, value in entry.items():
            if isinstance(value, str):
                count = value.lower().count("unknown")
                if count > 0:
                    total += count
                    detailed_hits.append((idx, key, count, value[:80]))

    return total, detailed_hits


if __name__ == "__main__":
    data = load_predictions(FILE)
    total, hits = count_unknown_words(data)

    print(f"Total 'unknown' WORD occurrences: {total}")
    print(f"Entries containing 'unknown': {len(hits)}\n")

    for idx, key, count, preview in hits[:20]:
        print(f"[{idx}] key={key}, count={count}, preview={preview}...")
