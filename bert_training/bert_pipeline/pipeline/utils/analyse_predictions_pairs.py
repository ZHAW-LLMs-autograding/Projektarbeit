import json
from collections import Counter

FILE_A = "/home/sercan/asag_pipeline/pipeline/metriken/benchmarks/bert/metrics_deduplicate.json"
FILE_B = "/home/sercan/projektarbeit/pipeline/metriken/benchmarks/bert/metrics_deduplicate.json"


def load(path):
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)
    if isinstance(data, dict) and "predictions" in data:
        return data["predictions"]
    return data


def make_key(entry):
    # Matching-Schlüssel NUR basierend auf den eigentlichen ASAG-Inhalten
    return (
        (entry.get("question") or "").strip(),
        (entry.get("reference_answer") or "").strip(),
        (entry.get("provided_answer") or "").strip(),
        str(entry.get("y_true")),
    )


def main():
    A = load(FILE_A)
    B = load(FILE_B)

    print(f"🔍 Datei A: {FILE_A} — {len(A)} Einträge")
    print(f"🔍 Datei B: {FILE_B} — {len(B)} Einträge\n")

    # ------------------------
    # UNKNOWN COUNTS
    # ------------------------
    unk_A = sum(1 for e in A if "unknown" in str(e).lower())
    unk_B = sum(1 for e in B if "unknown" in str(e).lower())

    print(f"❓ Unknown Vorkommen A: {unk_A}")
    print(f"❓ Unknown Vorkommen B: {unk_B}\n")

    # ------------------------
    # KEY-SETS vergleichen
    # ------------------------
    set_A = {make_key(e) for e in A}
    set_B = {make_key(e) for e in B}

    missing_in_B = set_A - set_B
    missing_in_A = set_B - set_A

    print("🔍 Abgleich der Items (per content-based key):")
    print(f"   → Nur in A: {len(missing_in_B)}")
    print(f"   → Nur in B: {len(missing_in_A)}\n")

    if missing_in_B:
        print("❌ Items nur in A (nicht in B):")
        for k in list(missing_in_B)[:5]:
            print("  ", k)
        print()

    if missing_in_A:
        print("❌ Items nur in B (nicht in A):")
        for k in list(missing_in_A)[:5]:
            print("  ", k)
        print()

    # ------------------------
    # Duplikate finden
    # ------------------------
    print("🔎 Prüfe auf Duplikate...")

    dup_A = [k for k, c in Counter(make_key(e) for e in A).items() if c > 1]
    dup_B = [k for k, c in Counter(make_key(e) for e in B).items() if c > 1]

    print(f"   → Duplikate in A: {len(dup_A)}")
    print(f"   → Duplikate in B: {len(dup_B)}\n")

    # ------------------------
    # Detaillierte Feldunterschiede
    # ------------------------
    print("🧪 Vergleiche identische Items Feld-für-Feld...\n")

    A_map = {make_key(e): e for e in A}
    B_map = {make_key(e): e for e in B}

    diff_count = 0

    for key in set_A & set_B:
        eA = A_map[key]
        eB = B_map[key]

        # Wir vergleichen ALLE Felder
        for field in eA:
            if field in ["y_pred", "raw_output_text", "inference_time_ms"]:
                # Diese Felder dürfen je Modell anders sein → ignorieren
                continue
            if eA[field] != eB.get(field):
                diff_count += 1
                print("⚠️ Unterschied in Feld:", field)
                print("   KEY:", key)
                print("   A:", eA[field])
                print("   B:", eB.get(field))
                print()

    print(f"📊 Gesamt unterschiedliche relevante Felder: {diff_count}")


if __name__ == "__main__":
    main()
