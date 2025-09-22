# src/train_smm4h_classification.py
import os
os.environ["WANDB_DISABLED"] = "true"
import numpy as np
from datasets import load_from_disk
from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
    Trainer,
    TrainingArguments,
    DataCollatorWithPadding,
)
from sklearn.metrics import precision_recall_fscore_support, accuracy_score

# ============== Load dataset ==================
data_path = "data/processed/hf_smm4h_subtask3"
dataset = load_from_disk(data_path)

# Labels
labels = sorted(set(dataset["train"]["label"]) | set(dataset["test"]["label"]))
label2id = {str(label): i for i, label in enumerate(labels)}
id2label = {i: str(label) for label, i in label2id.items()}

# ============== Tokenizer =====================
model_name = "dmis-lab/biobert-v1.1"
tokenizer = AutoTokenizer.from_pretrained(model_name)

def preprocess_function(examples):
    return tokenizer(
        examples["mention"],
        truncation=True,
        padding="max_length",
        max_length=64,
    )

tokenized = dataset.map(preprocess_function, batched=True)

# 🔧 Convert string labels -> integer IDs
def encode_labels(example):
    return {"labels": label2id[str(example["label"])]}

tokenized = tokenized.map(encode_labels)
tokenized = tokenized.remove_columns(["label"])  # drop original string label

print("Sample after processing:", tokenized["train"][0])  # debug

# ============== Model =========================
model = AutoModelForSequenceClassification.from_pretrained(
    model_name,
    num_labels=len(label2id),
    id2label=id2label,
    label2id=label2id,
)

# ============== Metrics =======================
def compute_metrics(p):
    preds = np.argmax(p.predictions, axis=1)
    labels = p.label_ids
    acc = accuracy_score(labels, preds)
    precision_macro, recall_macro, f1_macro, _ = precision_recall_fscore_support(
        labels, preds, average="macro", zero_division=0
    )
    precision_micro, recall_micro, f1_micro, _ = precision_recall_fscore_support(
        labels, preds, average="micro", zero_division=0
    )
    return {
        "accuracy": acc,
        "precision_macro": precision_macro,
        "recall_macro": recall_macro,
        "f1_macro": f1_macro,
        "precision_micro": precision_micro,
        "recall_micro": recall_micro,
        "f1_micro": f1_micro,
    }

# ============== Training ======================
args = TrainingArguments(
    output_dir="results/smm4h_classification_biobert",
    evaluation_strategy="epoch",
    save_strategy="epoch",
    learning_rate=2e-5,
    per_device_train_batch_size=16,
    per_device_eval_batch_size=16,
    num_train_epochs=10,
    weight_decay=0.01,
    logging_dir="logs",
    load_best_model_at_end=True,
    metric_for_best_model="f1_macro",
    greater_is_better=True,
)

if "evaluation_strategy" in TrainingArguments.__init__.__code__.co_varnames:
    training_args_kwargs["evaluation_strategy"] = "epoch"
else:
    training_args_kwargs["eval_strategy"] = "epoch"

args = TrainingArguments(**training_args_kwargs) 

data_collator = DataCollatorWithPadding(tokenizer=tokenizer)

trainer = Trainer(
    model=model,
    args=args,
    train_dataset=tokenized["train"],
    eval_dataset=tokenized["test"],  # ⚠️ test used as validation (no val set available)
    tokenizer=tokenizer,
    data_collator=data_collator,
    compute_metrics=compute_metrics,
)

trainer.train()
trainer.evaluate(tokenized["test"])
