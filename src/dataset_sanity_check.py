from datasets import load_from_disk
from collections import Counter
from pathlib import Path
import random

PROCESSED_DIR = Path("data/processed")

# ================
# CADEC CHECK
# ================
def check_cadec():
    print("\n🔎 Checking CADEC...")
    ds = load_from_disk(PROCESSED_DIR / "hf_cadec_reduced")
    print(ds)

    # Show one example
    example = random.choice(ds["train"])
    print("\nExample from CADEC:")
    print("Tokens:", example["tokens"])
    print("Labels:", example["labels"])

    # Label distribution
    all_labels = []
    for split in ["train", "validation", "test"]:
        all_labels.extend([lab for seq in ds[split]["labels"] for lab in seq])

    counts = Counter(all_labels)
    print("\nCADEC Label Distribution:")
    for k, v in counts.items():
        print(f"{k}: {v}")


# ================
# SMM4H CHECK
# ================

 
import json
def check_smm4h():
    print("\n🔎 Checking SMM4H Subtask3...")
    ds = load_from_disk(PROCESSED_DIR / "hf_smm4h_subtask3")
    print(ds)

    # Show one random example
    example = random.choice(ds["train"])
    print("\nExample from SMM4H Subtask3:")
    print(example)

    # Load MedDRA ID → term mapping from labels.json (if available)
    labels_map_path = Path("data/raw/smm4h_2017/labels.json")
    id_to_term = {}
    if labels_map_path.exists():
        try:
            with open(labels_map_path, "r", encoding="utf-8") as f:
                data = json.load(f)
                # handle two possible formats: list of dicts or dict directly
                if isinstance(data, dict):
                    id_to_term = data
                elif isinstance(data, list):
                    # Sometimes it’s list of tuples or list of dicts
                    id_to_term = {str(item[0]): item[1] if isinstance(item, list) else None for item in data}
            print(f"✅ Loaded {len(id_to_term)} label mappings from labels.json")
        except Exception as e:
            print(f"⚠️ Could not parse labels.json: {e}")

    # Collect all labels and mentions
    train_labels = list(ds["train"]["label"])
    test_labels  = list(ds["test"]["label"])
    labels = train_labels + test_labels

    # Build a mapping: MedDRA code -> most common mention
    code_to_mentions = {}
    for split in ["train", "test"]:
        for ex in ds[split]:
            code = ex["label"]
            mention = ex["mention"]
            if code not in code_to_mentions:
                code_to_mentions[code] = []
            code_to_mentions[code].append(mention)
            
    # Collapse to most common mention
    code_to_name = {c: Counter(m).most_common(1)[0][0] for c, m in code_to_mentions.items()}

    # Count frequencies
    counts = Counter(labels)

    print("\nSMM4H Label Distribution (train+test):")
    for k, v in counts.most_common(10):
        concept_name = code_to_name.get(k, "N/A")
        print(f"{k} ({concept_name}): {v}")

    print(f"\nTotal unique MedDRA codes: {len(set(labels))}")


# ================
# MAIN
# ================
if __name__ == "__main__":
    check_cadec()
    check_smm4h()
