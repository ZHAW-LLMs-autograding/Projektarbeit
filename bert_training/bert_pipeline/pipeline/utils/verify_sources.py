import json

OLD = "/home/sercan/asag_pipeline/pipeline/metriken/benchmarks/vectormodels/char_wb_3_5_min2/predictions.json"
NEW = "/home/sercan/asag_pipeline/pipeline/metriken/benchmarks/vectormodels/char_wb_3_5_min2/predictions_repaired.json"

def load(path):
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)

def compare_entries(old, new):
    """Vergleich aller Felder außer 'source'."""
    diffs = []
    for key in old:
        if key == "source":
            continue
        if old[key] != new[key]:
            diffs.append(key)
    return diffs


def main():
    old_data = load(OLD)
    new_data = load(NEW)

    print(f"Alte Einträge: {len(old_data)}")
    print(f"Neue Einträge: {len(new_data)}\n")

    if len(old_data) != len(new_data):
        print("❌ FEHLER: Unterschiedliche Anzahl an Einträgen!")
        return

    mismatches = 0
    source_changes = 0

    for i, (old_e, new_e) in enumerate(zip(old_data, new_data)):

        diffs = compare_entries(old_e, new_e)

        if diffs:
            mismatches += 1
            print(f"\n❌ Unterschied in Entry {i}: {diffs}")
            print("   Alte Werte:", {k: old_e[k] for k in diffs})
            print("   Neue Werte:", {k: new_e[k] for k in diffs})

        # Quelle sollte geändert worden sein
        if old_e["source"] != new_e["source"]:
            source_changes += 1

    print("\n====================")
    print("🔍 Vergleich abgeschlossen:")
    print(f"   → Unterschiede in Datenfeldern (NICHT source): {mismatches}")
    print(f"   → Geänderte source-Felder: {source_changes}")

    if mismatches == 0:
        print("🎉 OK: Alle Einträge sind 1:1 identisch, außer 'source'.")
    else:
        print("⚠️ Warnung: Es wurden Unterschiede entdeckt.")


if __name__ == "__main__":
    main()
