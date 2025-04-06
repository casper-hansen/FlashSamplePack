import os
import math
import hashlib
from pathlib import Path
from trl import SFTTrainer, SFTConfig
from datasets import load_dataset, Dataset, load_from_disk
from torch.utils.data import RandomSampler
from transformers import AutoTokenizer, PreTrainedTokenizer
from flashpack import (
    patch_for_multipack,
    qwen25_template,
    V2BatchSamplerDataCollatorForSeq2Seq,
    MultipackBatchSampler,
    prepare_dataset,
    get_dataset_lengths,
    cache_dataset,
)

OUTPUT_DIR = "./outputs"
DATASET_PREPARED_PATH = "./prepared_datasets"
DATASET_PATH = "HuggingFaceTB/smoltalk"
DATASET_NAME = "everyday-conversations"
DATASET_SPLIT = "train"
DATASET_COLUMN = "messages"
CHAT_TEMPLATE = qwen25_template
TRAIN_MICRO_BATCH_SIZE = 1
MODEL_PATH = "Qwen/Qwen2.5-1.5B-Instruct"
MIN_LEN = 32
MAX_LEN = 2048
FINGERPRINT_HASH = hashlib.md5(
     f"{DATASET_PATH}:{DATASET_NAME}:{DATASET_SPLIT}:{MIN_LEN}:{MAX_LEN}:{CHAT_TEMPLATE}".encode()
).hexdigest()
PREPARED_HASH_PATH = Path(DATASET_PREPARED_PATH) / FINGERPRINT_HASH
os.makedirs(PREPARED_HASH_PATH, exist_ok=True)


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


if __name__ == "__main__":
    tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH)
    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "left"

    if PREPARED_HASH_PATH.exists() and any(PREPARED_HASH_PATH.glob("*")):
        dataset = load_from_disk(str(PREPARED_HASH_PATH))
    else:
        dataset = load_dataset(DATASET_PATH, DATASET_NAME, split=DATASET_SPLIT)
        dataset = apply_chat_template(dataset, tokenizer, CHAT_TEMPLATE)
        dataset = prepare_dataset(dataset, MIN_LEN, MAX_LEN, {"num_proc": 8})
        dataset = cache_dataset(dataset, PREPARED_HASH_PATH)
    
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
            num_train_epochs=1,
            save_strategy="epoch",
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
            # gradient_checkpointing=True,
            # gradient_checkpointing_kwargs={"use_reentrant": False},
            model_init_kwargs={
                "attn_implementation": "flash_attention_2",
                "torch_dtype": "bfloat16",
            },
        ),
    )
    trainer.train()
