Perfect 👍 Let’s draft a **clear and professional `README.md`** for your GitHub repository. This will explain your thesis project, its purpose, datasets, methods, and how others can reproduce your results.

Here’s a first version:

---

# Thesis\_ADE\_Extraction

## 📖 Project Overview

This repository contains the implementation for my MSc Thesis:
**"Interpretable Extraction of Adverse Drug Events (ADEs): Comparing Biomedical Transformers and Large Language Models"**

The project investigates how **biomedical transformer models** (BioBERT, PubMedBERT) compare with **large language models** (BioGPT, GPT-based prompting) in extracting **Adverse Drug Events (ADEs)** from benchmark datasets.

---

## 🎯 Research Question

**RQ2:**
How do biomedical transformer models (BioBERT, PubMedBERT) compare with Large Language Models (e.g., BioGPT, GPT-based prompting) in extracting ADEs from benchmark datasets (CADEC, SMM4H)?
Can LLMs provide competitive or superior results in few-shot and zero-shot settings?

---

## 📌 Objectives

1. Implement and evaluate baseline transformer models (BioBERT, PubMedBERT) for:

   * **ADE Named Entity Recognition (NER)** – CADEC dataset
   * **ADE Normalization/Classification** – SMM4H dataset
2. Investigate LLM-based methods (prompting, few-shot, and fine-tuning BioGPT).
3. Compare performance across datasets using standardized metrics.
4. Conduct error analysis of transformer vs. LLM performance.
5. *(Optional)* Explore interpretability (attention visualization, SHAP).

---

## 📂 Repository Structure

```
Thesis_ADE_Extraction/
│
├── configs/                     # Config files
├── data/
│   ├── raw/                     # Original datasets
│   ├── processed/
│   │   ├── hf_cadec_reduced/    # HuggingFace-format CADEC
│   │   └── hf_smm4h_subtask3/   # HuggingFace-format SMM4H
│
├── figures/                     # Plots and visualizations
├── logs/                        # Training logs
├── models/                      # Saved transformer/LLM models
├── predictions/                 # Model outputs
│
├── src/
│   ├── models/                  # Model modules
│   │   ├── llm_prompting.py     # Few-shot/zero-shot prompting (LLMs)
│   │   ├── relation_extraction.py # Drug–ADE relation extraction
│   │   └── transformer_ner.py   # Transformer baseline (NER)
│   ├── data_preprocessing.py    # Preprocess raw datasets
│   ├── dataset_loader.py        # Load into HF format
│   ├── dataset_sanity_check.py  # Verify label/token consistency
│   ├── evaluate.py              # Evaluation utilities
│   ├── tokenizer_utils.py       # Align labels with tokens
│   ├── train_cadec_ner.py       # Train CADEC NER baseline
│   └── train_smm4h_classification.py # Train SMM4H classification baseline
│
├── thesis-paper/                # Drafts and reports
├── environment.yml              # Conda environment
├── README.md                    # Project description
└── .gitignore                   # Ignore large/log files
```

---

## ⚙️ Installation

Clone the repository and install dependencies:

```bash
git clone https://github.com/<your-username>/Thesis_ADE_Extraction.git
cd Thesis_ADE_Extraction
conda env create -f environment.yml
conda activate ade-thesis
```

---

## 📊 Datasets

* **CADEC** – Consumer Adverse Drug Effect Corpus (NER, entity-level ADE extraction).
* **SMM4H Subtask 3** – Social Media Mining for Health dataset (ADE normalization/classification).

Both are preprocessed into **HuggingFace format** in `data/processed/`.

---

## 🚀 Training & Evaluation

### 1. CADEC (NER with BioBERT)

```bash
python src/train_cadec_ner.py
```

### 2. SMM4H (Classification with BioBERT)

```bash
python src/train_smm4h_classification.py
```

Logs and model checkpoints are saved in `logs/` and `models/`.

---

## 📈 Results (so far)

* **CADEC NER (BioBERT):** strong micro-F1 ≈ **0.91** (baseline established).
* **SMM4H Classification (BioBERT, 10 epochs):** accuracy ≈ **85%**, macro-F1 ≈ **0.42**.

Next: PubMedBERT baselines + LLM prompting.

---

## 🏁 Roadmap

* [x] Baseline transformers (BioBERT on CADEC + SMM4H)
* [ ] PubMedBERT baselines
* [ ] LLM prompting experiments (GPT, BioGPT)
* [ ] Relation extraction module
* [ ] Error analysis & interpretability
* [ ] Write-up & defense preparation

---

## 📜 Expected Contributions

* Systematic comparison of **biomedical transformers vs. LLMs** for pharmacovigilance.
* Insights into whether **LLMs can replace or complement** domain-specific models.
* Transparent and interpretable ADE extraction pipeline.

---

👉 This is a clean, professional README. You can update the **Results** section as you run more experiments.

Would you like me to also include **example outputs** (like sample predictions from CADEC/SMM4H) in the README so it looks more illustrative?
