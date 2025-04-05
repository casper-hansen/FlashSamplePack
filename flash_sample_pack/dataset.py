from functools import partial
from typing import Dict
from datasets import Dataset
import numpy as np

def get_dataset_lengths(dataset, from_arrow=False):
    if "length" in dataset.column_names:
        lengths = np.array(dataset["length"])
    elif "position_ids" in dataset.column_names:
        position_ids = dataset["position_ids"]
        lengths = np.array([x[-1] + 1 for x in position_ids])
    else:
        if from_arrow:
            input_ids = dataset.data.column("input_ids")
            lengths = np.vectorize(len)(np.array(input_ids, dtype=object))
        else:
            input_ids = dataset["input_ids"]
            lengths = np.array([len(seq) for seq in input_ids])
    return lengths

def drop_sequences(sample, max_seq_len=2048, min_seq_len=2):
    """
    Drop samples whose sequence length is either too long (> sequence_len)
    or too short (< min_sequence_len).

    Works for both single-example (list[int]) or batched (list[list[int]]).
    """
    min_seq_len = min_seq_len or 2

    input_ids = sample["input_ids"]

    # Edge case: if input_ids is empty
    if not input_ids:
        # Decide if you want to drop or keep empty. Let's drop.
        return False

    # Check if single example or batched by looking at the first element
    if isinstance(input_ids[0], int):
        # Single example (input_ids is a list of int)
        length = len(input_ids)
        return min_seq_len <= length <= max_seq_len

    # Batched (input_ids is a list of lists)
    results = []
    for seq in input_ids:
        length = len(seq)
        results.append(min_seq_len <= length <= max_seq_len)
    return results


def drop_no_trainable_tokens(sample):
    """
    Drop samples if all labels are -100 (i.e., zero trainable tokens).
    Works for both single-example or batched input.
    """
    labels = sample["labels"]
    if not labels:
        return True

    # Check if single example or batch
    # If first element is an int, we assume a single example
    # If it's a list, we assume we're dealing with a batch
    if isinstance(labels[0], int):
        # Single example: return a single bool
        return np.any(labels != -100)

    # Batched: 'labels' is a list of lists
    # Return a list of booleans, one per sub-list
    results = [np.any(row_labels != -100) for row_labels in labels]
    return results


def add_position_ids(sample):
    """
    Handle both single-example and batched data.
    - single example: sample['input_ids'] is a list[int]
    - batched data: sample['input_ids'] is a list[list[int]]
    """
    # Return sample unchanged if "input_ids" is not present, or is empty
    if "input_ids" not in sample or not sample["input_ids"]:
        return sample

    input_ids = sample["input_ids"]

    # If first element is an int, it’s a single example
    # If first element is a list, it’s a batch
    if isinstance(input_ids[0], int):
        # ---- SINGLE EXAMPLE ----
        seq_len = len(input_ids)
        # Position IDs for a single example
        # As a list
        sample["position_ids"] = list(range(seq_len))
        sample["length"] = seq_len

    else:
        # ---- BATCHED EXAMPLES ----
        # input_ids is a list of lists
        position_ids_batch = []
        lengths_batch = []
        for seq in input_ids:
            seq_len = len(seq)
            position_ids_batch.append(list(range(seq_len)))
            lengths_batch.append(seq_len)

        # Now store them back
        sample["position_ids"] = position_ids_batch
        sample["length"] = lengths_batch

    return sample


def prepare_dataset(
    dataset: Dataset, filter_map_kwargs: Dict = {}, min_seq_len=2, max_seq_len=2048
):
    drop_long_short_seq = partial(
        drop_sequences,
        max_seq_len=max_seq_len,
        min_seq_len=min_seq_len,
    )

    dataset = dataset.filter(
        drop_long_short_seq,
        batched=True,
        desc=f"Dropping Long (>={max_seq_len}) and Short (<={min_seq_len}) Sequences",
        **filter_map_kwargs,
    )

    dataset = dataset.map(
        drop_no_trainable_tokens,
        batched=True,
        desc="Drop Samples with Zero Trainable Tokens",
        **filter_map_kwargs,
    )

    dataset = dataset.map(
        add_position_ids,
        batched=True,
        desc="Add position_id column (Sample Packing)",
        **filter_map_kwargs,
    )

    return dataset
