# src/train_cadec_ner.py
import os
os.environ["WANDB_DISABLED"] = "true"

import numpy as np
import torch
from datasets import load_from_disk
from transformers import (
    AutoTokenizer,
    AutoModelForTokenClassification,
    Trainer,
    TrainingArguments,
    DataCollatorForTokenClassification,
    EarlyStoppingCallback,
)
from sklearn.metrics import precision_recall_fscore_support
from sklearn.model_selection import KFold
from tokenizer_utils import align_labels_with_tokens

# ============== Load dataset ==================
data_path = "data/processed/hf_cadec_reduced"
dataset = load_from_disk(data_path)

# Labels
labels = sorted(set(l for seq in dataset["train"]["labels"] for l in seq))
label2id = {label: i for i, label in enumerate(labels)}
id2label = {i: label for label, i in label2id.items()}

# Tokenizer
model_name = "dmis-lab/biobert-v1.1"
tokenizer = AutoTokenizer.from_pretrained(model_name)

def tokenize_and_align(examples):
    return align_labels_with_tokens(
        examples["tokens"], examples["labels"], tokenizer, label2id, max_length=256
    )

tokenized = dataset.map(tokenize_and_align, batched=True)

# Compute class weights for ADR/Drug imbalance
all_labels = [l for seq in dataset["train"]["labels"] for l in seq if l != "O"]
classes, counts = np.unique(all_labels, return_counts=True)
class_weights = {label2id[c]: 1.0 / count for c, count in zip(classes, counts)}

# ============== Model wrapper =================
from torch import nn

class WeightedTrainer(Trainer):
    def compute_loss(self, model, inputs, return_outputs=False):
        labels = inputs.get("labels")
        outputs = model(**inputs)
        logits = outputs.get("logits")
        loss_fct = nn.CrossEntropyLoss(
            weight=torch.tensor([class_weights.get(i, 1.0) for i in range(len(label2id))]).to(model.device),
            ignore_index=-100,
        )
        loss = loss_fct(logits.view(-1, model.config.num_labels), labels.view(-1))
        return (loss, outputs) if return_outputs else loss

# ============== Metrics =======================
def compute_metrics(p):
    preds = np.argmax(p.predictions, axis=2)
    true_labels = p.label_ids
    preds_flat, labels_flat = [], []
    for pred, label in zip(preds, true_labels):
        for p_, l_ in zip(pred, label):
            if l_ != -100:
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
    learning_rate=2e-5,
    per_device_train_batch_size=8,
    per_device_eval_batch_size=8,
    num_train_epochs=10,
    weight_decay=0.01,
    logging_dir="logs",
    load_best_model_at_end=True,
    metric_for_best_model="f1_macro",
    greater_is_better=True,
    save_total_limit=2,
    evaluation_strategy="epoch",
    save_strategy="epoch",
)

data_collator = DataCollatorForTokenClassification(tokenizer)

# Single run (default)
model = AutoModelForTokenClassification.from_pretrained(
    model_name, num_labels=len(label2id), id2label=id2label, label2id=label2id
)
trainer = WeightedTrainer(
    model=model,
    args=args,
    train_dataset=tokenized["train"],
    eval_dataset=tokenized["validation"],
    tokenizer=tokenizer,
    data_collator=data_collator,
    compute_metrics=compute_metrics,
    callbacks=[EarlyStoppingCallback(early_stopping_patience=2)],
)
trainer.train()
trainer.evaluate(tokenized["test"])

# Optional: K-Fold CV
# kf = KFold(n_splits=5, shuffle=True, random_state=42)
# for fold, (train_idx, val_idx) in enumerate(kf.split(tokenized["train"])):
#     ...
