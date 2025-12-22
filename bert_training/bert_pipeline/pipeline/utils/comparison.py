import json
import sys
from pathlib import Path

def load_json(path):
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)

def main(file1, file2):
    p1 = Path(file1)
    p2 = Path(file2)

    if not p1.exists() or not p2.exists():
        print("❌ Eine der Dateien existiert nicht.")
        return

    j1 = load_json(p1)
    j2 = load_json(p2)

    if j1 == j2:
        print("✔️ Die JSON-Dateien sind *identisch*.")
    else:
        print("❌ Die JSON-Dateien unterscheiden sich!")
        print("\n--- Unterschiede (Debug) ---")
        print_diff(j1, j2)

def print_diff(j1, j2, path="root"):
    if type(j1) != type(j2):
        print(f"Typunterschied bei {path}: {type(j1)} != {type(j2)}")
        return

    # Dictionaries vergleichen
    if isinstance(j1, dict):
        keys1 = set(j1.keys())
        keys2 = set(j2.keys())

        for k in keys1 - keys2:
            print(f"{path}: Key nur in Datei 1: {k}")
        for k in keys2 - keys1:
            print(f"{path}: Key nur in Datei 2: {k}")

        for k in keys1 & keys2:
            print_diff(j1[k], j2[k], f"{path}.{k}")
        return

    # Listen vergleichen
    if isinstance(j1, list):
        if len(j1) != len(j2):
            print(f"{path}: Listenlänge unterschiedlich: {len(j1)} != {len(j2)}")

        for i, (a, b) in enumerate(zip(j1, j2)):
            print_diff(a, b, f"{path}[{i}]")
        return

    # Primitive: direkter Vergleich
    if j1 != j2:
        print(f"{path}: Werte unterschiedlich: {j1} != {j2}")

if __name__ == "__main__":
    if len(sys.argv) != 3:
        print("Benutzung: python compare_json.py file1.json file2.json")
        sys.exit(1)

    main(sys.argv[1], sys.argv[2])
