# data_preprocessing.py
import json
import random
from collections import Counter
from datasets import Dataset, DatasetDict, load_dataset
from pathlib import Path

random.seed(42)

# =========================
# Global paths
# =========================
RAW_DIR = Path("data/raw")
PROCESSED_DIR = Path("data/processed")
PROCESSED_DIR.mkdir(parents=True, exist_ok=True)

# =========================
# CADEC Preprocessing
# =========================
def preprocess_cadec():
    DATASET_NAME = "akramRedjdal/cadec-ner-dataset-clean"
    print(f"\n⬇️  Loading CADEC dataset from Hugging Face: {DATASET_NAME}")
    ds = load_dataset(DATASET_NAME)

    # Keep ADR + Drug, map others to O
    MAP_TO = {
        "B-ADR": "B-ADR", "I-ADR": "I-ADR",
        "B-Drug": "B-Drug", "I-Drug": "I-Drug"
    }

    def map_labels(label_seq):
        return [MAP_TO.get(l, "O") for l in label_seq]

    def convert_split(split_dataset):
        return [{"tokens": ex["tokens"], "labels": map_labels(ex["labels"])} for ex in split_dataset]

    # Train/dev split
    train_list = convert_split(ds["train"])
    test_list = convert_split(ds["test"])
    random.shuffle(train_list)
    n_train = int(0.8 * len(train_list))
    train_proc, dev_proc = train_list[:n_train], train_list[n_train:]

    # Convert to HF datasets
    hf_train = Dataset.from_dict({"tokens": [ex["tokens"] for ex in train_proc],
                                  "labels": [ex["labels"] for ex in train_proc]})
    hf_dev = Dataset.from_dict({"tokens": [ex["tokens"] for ex in dev_proc],
                                "labels": [ex["labels"] for ex in dev_proc]})
    hf_test = Dataset.from_dict({"tokens": [ex["tokens"] for ex in test_list],
                                 "labels": [ex["labels"] for ex in test_list]})

    ds_out = DatasetDict({"train": hf_train, "validation": hf_dev, "test": hf_test})
    out_path = PROCESSED_DIR / "hf_cadec_reduced"
    ds_out.save_to_disk(out_path)

    print(f"✅ Saved CADEC to {out_path}")
    return ds_out

# =========================
# SMM4H Subtask3: Concept Normalization
# =========================
from datasets import Dataset, DatasetDict
from collections import Counter

def preprocess_smm4h_subtask3(min_freq: int = 5):
    base_path = RAW_DIR / "smm4h_2017" / "subtask3"

    # Collect all training files
    train_files = [
        base_path / "task_3_normalization_training1.txt",
        base_path / "task_3_normalization_training2.txt",
        base_path / "task_3_normalization_training3.txt",
        base_path / "task_3_normalization_training4.txt",
    ]
    eval_file = base_path / "task_3_normalization_evaluation.txt"

    # Parse TSV-like text files
    def parse_file(path):
        examples = []
        with open(path, "r", encoding="utf-8") as f:
            for line in f:
                parts = line.strip().split("\t")
                if len(parts) != 3:
                    continue
                ex_id, mention, label = parts
                examples.append({
                    "id": int(ex_id),
                    "mention": mention,
                    "label": label   # MedDRA code as string
                })
        return examples

    # Load training and eval
    train_examples = []
    for f in train_files:
        if f.exists():
            train_examples.extend(parse_file(f))
        else:
            print(f"⚠️ Missing file: {f}")

    eval_examples = parse_file(eval_file) if eval_file.exists() else []

    # Count frequencies in training
    label_counts = Counter([ex["label"] for ex in train_examples])
    rare_labels = {lbl for lbl, cnt in label_counts.items() if cnt < min_freq}

    # Replace rare codes with "OTHER"
    def replace_rare(ex):
        if ex["label"] in rare_labels:
            ex["label"] = "OTHER"
        return ex

    train_examples = [replace_rare(ex) for ex in train_examples]
    eval_examples = [replace_rare(ex) for ex in eval_examples]

    # Convert to HF dataset
    hf_train = Dataset.from_list(train_examples)
    hf_eval = Dataset.from_list(eval_examples)

    ds_out = DatasetDict({
        "train": hf_train,
        "test": hf_eval
    })

    # Save
    out_dir = PROCESSED_DIR / "hf_smm4h_subtask3"
    ds_out.save_to_disk(out_dir)
    print(f"✅ Saved SMM4H Subtask3 (with rare codes grouped to OTHER) to {out_dir}")

    # Stats
    print(f"Unique labels after grouping: {len(set(hf_train['label']))}")
    print("Top 10 labels:", Counter(hf_train["label"]).most_common(10))

    return ds_out

# =========================
# MAIN
# =========================
if __name__ == "__main__":
    # Run CADEC preprocessing
    print("\nProcessing cadec..")
    cadec_ds = preprocess_cadec()

    # Run SMM4H Subtask2 preprocessing
    print("\nProcessing SMM4H..")
    smm4h_ds = preprocess_smm4h_subtask3()
    
    

