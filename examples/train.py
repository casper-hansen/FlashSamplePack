import math
import os
import hashlib
from pathlib import Path
from trl import SFTTrainer, SFTConfig
from datasets import load_dataset, Dataset, load_from_disk
from torch.utils.data import RandomSampler
from transformers import AutoTokenizer, PreTrainedTokenizer
from flash_sample_pack import (
    patch_for_multipack,
    qwen25_template,
    V2BatchSamplerDataCollatorForSeq2Seq,
    MultipackBatchSampler,
    prepare_dataset,
    get_dataset_lengths,
)

OUTPUT_DIR = "./outputs"
DATASET_PREPARED_PATH = "./prepared_datasets"
DATASET_PATH = "HuggingFaceTB/smoltalk"
DATASET_NAME = "everyday-conversations"
DATASET_SPLIT = "train"
DATASET_COLUMN = "messages"
TRAIN_MICRO_BATCH_SIZE = 1
MODEL_PATH = "Qwen/Qwen2.5-1.5B-Instruct"
MIN_LEN = 32
MAX_LEN = 2048


def apply_chat_template(
    dataset: Dataset,
    tokenizer: PreTrainedTokenizer,
    chat_template: str,
) -> Dataset:
    def map_fn(example):
        formatted_chat = tokenizer.apply_chat_template(
            example[DATASET_COLUMN],
            chat_template=chat_template,
            tokenize=True,
            add_generation_prompt=False,
            return_dict=True,
        )
        labels = formatted_chat["input_ids"].copy()

        # Predict every token after the first one
        formatted_chat["labels"] = [-100] + labels[1:]

        return formatted_chat

    dataset = dataset.map(
        map_fn,
        num_proc=8,
        desc="Applying Chat Template",
        remove_columns=[DATASET_COLUMN],
    )

    tokenizer.chat_template = chat_template

    return dataset

def md5(text):
    """Generate MD5 hash for a string."""
    return hashlib.md5(text.encode()).hexdigest()

def load_or_prepare_dataset(
    tokenizer: PreTrainedTokenizer,
    chat_template: str,
    dataset_path: str = DATASET_PATH,
    dataset_name: str = DATASET_NAME,
    dataset_split: str = DATASET_SPLIT,
    min_len: int = MIN_LEN,
    max_len: int = MAX_LEN,
    dataset_prepared_path: str = DATASET_PREPARED_PATH,
):
    """Load a cached dataset or prepare and cache a new one."""
    ds_hash = md5(
        f"{dataset_path}:{dataset_name}:{dataset_split}:{min_len}:{max_len}:{chat_template}"
    )
    
    prepared_ds_path = Path(dataset_prepared_path) / ds_hash
    
    if prepared_ds_path.exists() and any(prepared_ds_path.glob("*")):
        print(f"Loading prepared dataset from disk at {prepared_ds_path}...")
        dataset = load_from_disk(str(prepared_ds_path))
        print("Prepared dataset loaded from disk...")
    else:
        print(f"Unable to find prepared dataset in {prepared_ds_path}")
        print("Loading and processing raw dataset...")
        
        dataset = load_dataset(dataset_path, dataset_name, split=dataset_split)
        dataset = apply_chat_template(dataset, tokenizer, chat_template)
        dataset = prepare_dataset(dataset, min_len, max_len, {"num_proc": 8})
        
        print(f"Saving prepared dataset to disk... {prepared_ds_path}")
        os.makedirs(prepared_ds_path, exist_ok=True)
        dataset.save_to_disk(str(prepared_ds_path))
    
    return dataset

if __name__ == "__main__":
    tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH)
    tokenizer.pad_token = tokenizer.eos_token

    dataset = load_or_prepare_dataset(
        tokenizer=tokenizer,
        chat_template=qwen25_template,
        dataset_path=DATASET_PATH,
        dataset_name=DATASET_NAME,
        dataset_split=DATASET_SPLIT,
        min_len=MIN_LEN,
        max_len=MAX_LEN,
    )

    batch_sampler = MultipackBatchSampler(
        RandomSampler(dataset),
        lengths=get_dataset_lengths(dataset),
        packing_efficiency_estimate=1.0,
        batch_max_len=TRAIN_MICRO_BATCH_SIZE * MAX_LEN,
        batch_size=TRAIN_MICRO_BATCH_SIZE,
        drop_last=True,
    )

    # NOTE: here we patch model and trainer internals in HF transformers
    patch_for_multipack(batch_sampler)

    collator = V2BatchSamplerDataCollatorForSeq2Seq(
        tokenizer=tokenizer,
        padding=True,
        max_length=MAX_LEN,
        pad_to_multiple_of=8 * math.ceil(MAX_LEN / 8),
    )

    trainer = SFTTrainer(
        model=MODEL_PATH,
        processing_class=tokenizer,
        train_dataset=dataset,
        data_collator=collator,
        args=SFTConfig(
            max_steps=200,
            output_dir=OUTPUT_DIR,
            dataset_text_field=None,
            max_seq_length=MAX_LEN,
            dataset_num_proc=8,
            per_device_train_batch_size=TRAIN_MICRO_BATCH_SIZE,
            include_tokens_per_second=True,
            gradient_accumulation_steps=1,
            learning_rate=2e-5,
            max_grad_norm=1.0,
            warmup_ratio=0.1,
            weight_decay=0.01,
            logging_steps=1,
            bf16=True,
            optim="adamw_torch_fused",
            gradient_checkpointing=True,
            gradient_checkpointing_kwargs={"use_reentrant": False},
            model_init_kwargs={
                "attn_implementation": "flash_attention_2",
                "torch_dtype": "bfloat16"
            },
        ),
    )
    trainer.train()
