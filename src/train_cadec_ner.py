# src/train_cadec_ner.py
import os
os.environ["WANDB_DISABLED"] = "true"
import numpy as np
from datasets import load_from_disk
from transformers import (
    AutoTokenizer,
    AutoModelForTokenClassification,
    Trainer,
    TrainingArguments,
    DataCollatorForTokenClassification,
)
from sklearn.metrics import precision_recall_fscore_support
from tokenizer_utils import align_labels_with_tokens

# ============== Load dataset ==================
data_path = "data/processed/hf_cadec_reduced"
dataset = load_from_disk(data_path)

# Labels
labels = sorted(set(l for seq in dataset["train"]["labels"] for l in seq))
label2id = {label: i for i, label in enumerate(labels)}
id2label = {i: label for label, i in label2id.items()}

# ============== Tokenizer =====================
model_name = "dmis-lab/biobert-v1.1"  # or try PubMedBERT, bert-base
tokenizer = AutoTokenizer.from_pretrained(model_name)

def tokenize_and_align(examples):
    return align_labels_with_tokens(
        examples["tokens"], examples["labels"], tokenizer, label2id, max_length=256
    )

tokenized = dataset.map(tokenize_and_align, batched=True)

# ============== Model =========================
model = AutoModelForTokenClassification.from_pretrained(
    model_name,
    num_labels=len(label2id),
    id2label=id2label,
    label2id=label2id,
)

# ============== Metrics =======================
def compute_metrics(p):
    preds = np.argmax(p.predictions, axis=2)
    true_labels = p.label_ids
    preds_flat, labels_flat = [], []
    for pred, label in zip(preds, true_labels):
        for p_, l_ in zip(pred, label):
            if l_ != -100:  # ignore padding
                preds_flat.append(p_)
                labels_flat.append(l_)
    precision_macro, recall_macro, f1_macro, _ = precision_recall_fscore_support(
        labels_flat, preds_flat, average="macro", zero_division=0
    )
    precision_micro, recall_micro, f1_micro, _ = precision_recall_fscore_support(
        labels_flat, preds_flat, average="micro", zero_division=0
    )
    return {
        "precision_macro": precision_macro,
        "recall_macro": recall_macro,
        "f1_macro": f1_macro,
        "precision_micro": precision_micro,
        "recall_micro": recall_micro,
        "f1_micro": f1_micro,
    }

# ============== Training ======================
args = TrainingArguments(
    output_dir="results/cadec_ner_biobert",
    evaluation_strategy="epoch",
    save_strategy="epoch",
    learning_rate=2e-5,
    per_device_train_batch_size=8,
    per_device_eval_batch_size=8,
    num_train_epochs=3,
    weight_decay=0.01,
    logging_dir="logs",
    load_best_model_at_end=True,
    metric_for_best_model="f1_macro",
    greater_is_better=True,
)

data_collator = DataCollatorForTokenClassification(tokenizer)

trainer = Trainer(
    model=model,
    args=args,
    train_dataset=tokenized["train"],
    eval_dataset=tokenized["validation"],
    tokenizer=tokenizer,
    data_collator=data_collator,
    compute_metrics=compute_metrics,
)

trainer.train()
trainer.evaluate(tokenized["test"])
