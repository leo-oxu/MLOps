import json
from pathlib import Path

INPUT = Path("data/raw/travel_data.json")
OUTPUT = Path("data/processed/cleaned_data.json")

def clean_tags(tags_str):
    return [t.strip().lower() for t in tags_str.split(",")]

def clean_entry(entry):
    return {
        "name": entry["name"].strip(),
        "country": entry["country"].strip(),
        "tags": [t.strip().lower() for t in entry["tags"]],
        "season": [s.strip().lower() for s in entry["season"]],
        "climate": entry["climate"].strip().lower(),
        "budget_level": entry["budget_level"].strip().lower(),
        "highlights": [h.strip() for h in entry["highlights"]]
    }

def main():
    with open(INPUT, "r", encoding="utf-8") as f:
        data = json.load(f)

    cleaned = [clean_entry(entry) for entry in data]

    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    with open(OUTPUT, "w", encoding="utf-8") as f:
        json.dump(cleaned, f, indent=2, ensure_ascii=False)

    print(f"✅ Cleaned {len(cleaned)} destinations → {OUTPUT}")

if __name__ == "__main__":
    main()