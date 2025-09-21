"""
dataset_loader.py

Utility functions to load CADEC (Adverse Drug Reaction) datasets
for training/evaluation. Handles both raw and processed versions.

Author: [Your Name]
"""

from pathlib import Path
from datasets import load_dataset, load_from_disk

# Paths relative to project root
RAW_PATH = Path("data/raw/cadec_raw")
PROCESSED_PATH = Path("data/processed/hf_cadec_reduced")

def load_raw_cadec():
    """
    Load the raw CADEC dataset.
    If not available locally in data/raw, it will download from Hugging Face
    and save it for reproducibility.
    """
    raw_path = RAW_PATH
    if raw_path.exists():
        print(f"✅ Loading CADEC raw dataset from {raw_path}")
        dataset = load_from_disk(str(raw_path))
    else:
        print("⬇️  Downloading CADEC raw dataset from Hugging Face...")
        dataset = load_dataset("akramRedjdal/cadec-ner-dataset-clean")
        dataset.save_to_disk(str(raw_path))
        print(f"💾 Saved raw dataset to {raw_path}")
    return dataset

def load_processed_cadec():
    """
    Load the processed CADEC dataset with reduced labels
    (ADR + Drug + O only).
    """
    proc_path = PROCESSED_PATH
    if not proc_path.exists():
        raise FileNotFoundError(
            f"❌ Processed dataset not found at {proc_path}. "
            "Run preprocessing script first!"
        )
    print(f"✅ Loading CADEC processed dataset from {proc_path}")
    dataset = load_from_disk(str(proc_path))
    return dataset

if __name__ == "__main__":
    # Demo run
    raw = load_raw_cadec()
    print(raw)

    processed = load_processed_cadec()
    print(processed)
