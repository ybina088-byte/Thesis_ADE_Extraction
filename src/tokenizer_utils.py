# src/tokenizer_utils.py
from typing import List, Dict
from transformers import PreTrainedTokenizerBase

def align_labels_with_tokens(
    texts: List[List[str]], 
    labels: List[List[str]], 
    tokenizer: PreTrainedTokenizerBase, 
    label2id: Dict[str, int],
    max_length: int = 256
):
    """
    Tokenize texts and align NER labels with subword tokens.

    Args:
        texts: list of tokenized sentences (list of words per sentence).
        labels: list of label sequences aligned with texts.
        tokenizer: Hugging Face tokenizer (BERT, BioBERT, etc).
        label2id: mapping from label string to integer ID.
        max_length: max sequence length for padding/truncation.

    Returns:
        Dictionary with input_ids, attention_mask, and aligned labels.
    """
    tokenized_inputs = tokenizer(
        texts,
        is_split_into_words=True,
        padding="max_length",
        truncation=True,
        max_length=max_length,
        return_tensors="pt"  # use PyTorch tensors
    )

    all_labels = []
    for i, label_seq in enumerate(labels):
        word_ids = tokenized_inputs.word_ids(batch_index=i)  # map tokens -> words
        aligned_labels = []
        prev_word_id = None
        for word_id in word_ids:
            if word_id is None:
                aligned_labels.append(-100)  # ignore token
            elif word_id != prev_word_id:
                aligned_labels.append(label2id[label_seq[word_id]])
            else:
                # Same word → subword, label it the same (or "I-" variant if needed)
                aligned_labels.append(label2id[label_seq[word_id]])
            prev_word_id = word_id
        all_labels.append(aligned_labels)

    tokenized_inputs["labels"] = all_labels
    return tokenized_inputs
