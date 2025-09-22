from typing import List, Dict
from transformers import PreTrainedTokenizerBase

def align_labels_with_tokens(
    texts: List[List[str]],
    labels: List[List[str]],
    tokenizer: PreTrainedTokenizerBase,
    label2id: Dict[str, int],
    max_length: int = 256,
    subword_strategy: str = "same"  # options: "same" or "iob2"
):
    """
    Tokenize texts and align NER labels with subword tokens.

    Args:
        texts: list of tokenized sentences (list of words per sentence).
        labels: list of label sequences aligned with texts.
        tokenizer: Hugging Face tokenizer (BioBERT, PubMedBERT, etc).
        label2id: mapping from label string to integer ID.
        max_length: max sequence length for padding/truncation.
        subword_strategy: how to handle subwords:
            - "same": copy the parent word's label
            - "iob2": convert subwords into I-XXX if parent is B-XXX

    Returns:
        dict with input_ids, attention_mask, and aligned labels.
    """
    tokenized_inputs = tokenizer(
        texts,
        is_split_into_words=True,
        padding="max_length",
        truncation=True,
        max_length=max_length,
        return_tensors=None,  # keep lists for Hugging Face Datasets
    )

    all_labels = []
    trunc_count = 0

    for i, label_seq in enumerate(labels):
        word_ids = tokenized_inputs.word_ids(batch_index=i)
        aligned_labels = []
        prev_word_id = None

        for word_id in word_ids:
            if word_id is None:
                aligned_labels.append(-100)
            else:
                raw_label = label_seq[word_id]
                if word_id != prev_word_id:
                    aligned_labels.append(label2id[raw_label])
                else:
                    if subword_strategy == "same":
                        aligned_labels.append(label2id[raw_label])
                    elif subword_strategy == "iob2":
                        if raw_label.startswith("B-"):
                            aligned_labels.append(label2id[raw_label.replace("B-", "I-")])
                        else:
                            aligned_labels.append(label2id[raw_label])
                prev_word_id = word_id

        # Track if truncation happened
        if len(aligned_labels) < max_length:
            trunc_count += 1

        all_labels.append(aligned_labels)

    if trunc_count > 0:
        print(f"⚠️ Warning: {trunc_count} sequences were truncated at {max_length} tokens.")

    tokenized_inputs["labels"] = all_labels
    return tokenized_inputs
