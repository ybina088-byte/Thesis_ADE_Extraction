# src/train_smm4h_classification.py
import os
os.environ["WANDB_DISABLED"] = "true"

import numpy as np
import torch
from datasets import load_from_disk, DatasetDict
from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
    Trainer,
    TrainingArguments,
    DataCollatorWithPadding,
    EarlyStoppingCallback,
)
from sklearn.metrics import precision_recall_fscore_support, accuracy_score
from torch import nn
from sklearn.model_selection import train_test_split

# Load dataset
data_path = "data/processed/hf_smm4h_subtask3"
dataset = load_from_disk(data_path)

# Create validation split
train_ds = dataset["train"]
train_split, val_split = train_test_split(train_ds, test_size=0.1, random_state=42, stratify=train_ds["label"])
dataset = DatasetDict({"train": train_split, "validation": val_split, "test": dataset["test"]})

# Labels
labels = sorted(set(dataset["train"]["label"]) | set(dataset["test"]["label"]))
label2id = {str(label): i for i, label in enumerate(labels)}
id2label = {i: str(label) for label, i in label2id.items()}

# Tokenizer
model_name = "dmis-lab/biobert-v1.1"
tokenizer = AutoTokenizer.from_pretrained(model_name)

def preprocess_function(examples):
    return tokenizer(examples["mention"], truncation=True, padding="max_length", max_length=64)

tokenized = dataset.map(preprocess_function, batched=True)

# Encode labels
def encode_labels(example):
    return {"labels": label2id[str(example["label"])]}

tokenized = tokenized.map(encode_labels)
tokenized = tokenized.remove_columns(["label"])

# Class weights (inverse frequency)
labels_all = tokenized["train"]["labels"]
classes, counts = np.unique(labels_all, return_counts=True)
class_weights = np.zeros(len(label2id))
for c, count in zip(classes, counts):
    class_weights[c] = 1.0 / count

# Model with weighted loss
class WeightedTrainer(Trainer):
    def compute_loss(self, model, inputs, return_outputs=False):
        labels = inputs.get("labels")
        outputs = model(**inputs)
        logits = outputs.get("logits")
        loss_fct = nn.CrossEntropyLoss(weight=torch.tensor(class_weights).to(model.device))
        loss = loss_fct(logits.view(-1, model.config.num_labels), labels.view(-1))
        return (loss, outputs) if return_outputs else loss

model = AutoModelForSequenceClassification.from_pretrained(
    model_name, num_labels=len(label2id), id2label=id2label, label2id=label2id
)

# Metrics
def compute_metrics(p):
    preds = np.argmax(p.predictions, axis=1)
    labels = p.label_ids
    acc = accuracy_score(labels, preds)
    precision_macro, recall_macro, f1_macro, _ = precision_recall_fscore_support(labels, preds, average="macro", zero_division=0)
    precision_micro, recall_micro, f1_micro, _ = precision_recall_fscore_support(labels, preds, average="micro", zero_division=0)
    return {
        "accuracy": acc,
        "precision_macro": precision_macro,
        "recall_macro": recall_macro,
        "f1_macro": f1_macro,
        "precision_micro": precision_micro,
        "recall_micro": recall_micro,
        "f1_micro": f1_micro,
    }

# Training
args = TrainingArguments(
    output_dir="results/smm4h_classification_biobert",
    learning_rate=2e-5,
    per_device_train_batch_size=16,
    per_device_eval_batch_size=16,
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

data_collator = DataCollatorWithPadding(tokenizer=tokenizer)

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
