import os
import time
import hashlib
from pathlib import Path
from filelock import FileLock
from trl import SFTTrainer, SFTConfig
from datasets import load_dataset, Dataset, load_from_disk
from torch.utils.data import RandomSampler
from transformers import AutoTokenizer, PreTrainedTokenizer
from flashpack import (
    patch_for_multipack,
    mistral_template,
    V2BatchSamplerDataCollatorForSeq2Seq,
    MultipackBatchSampler,
    prepare_dataset,
    get_dataset_lengths,
    cache_dataset,
    zero_first,
    is_local_main_process,
)

OUTPUT_DIR = "./outputs"
DATASET_PREPARED_PATH = "./prepared_datasets"
DATASET_PATH = "PatentPilotAI/patent-instruct-v3.1-length-sorted"
DATASET_NAME = None
DATASET_SPLIT = "train"
CHAT_TEMPLATE = mistral_template
TRAIN_MICRO_BATCH_SIZE = 1
MODEL_PATH = "mistralai/Mistral-Nemo-Base-2407"
MIN_LEN = 32
MAX_LEN = 65536
FINGERPRINT_HASH = hashlib.md5(
     f"{MODEL_PATH}:{DATASET_PATH}:{DATASET_NAME}:{DATASET_SPLIT}:{MIN_LEN}:{MAX_LEN}:{CHAT_TEMPLATE}".encode()
).hexdigest()
PREPARED_HASH_PATH = Path(DATASET_PREPARED_PATH) / FINGERPRINT_HASH
LOCK_FILE = PREPARED_HASH_PATH / ".prep.lock"
READY_FLAG = PREPARED_HASH_PATH / ".ready"
os.makedirs(PREPARED_HASH_PATH, exist_ok=True)


def apply_chat_template(
    dataset: Dataset,
    tokenizer: PreTrainedTokenizer,
    chat_template: str,
) -> Dataset:
    def map_fn(example):
        formatted_chat = tokenizer.apply_chat_template(
            example["conversation"],
            chat_template=chat_template,
            tokenize=True,
            return_dict=True,
        )
        formatted_chat["labels"] = formatted_chat["input_ids"].copy()

        return formatted_chat

    dataset = dataset.map(
        map_fn,
        num_proc=64,
        desc="Applying Chat Template",
    )

    tokenizer.chat_template = chat_template

    return dataset


if __name__ == "__main__":
    tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH)
    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "left"

    with FileLock(str(LOCK_FILE)):
        # rank-0 (the one that acquires the lock first) does the heavy work
        if not READY_FLAG.exists():
            dataset = load_dataset(DATASET_PATH, DATASET_NAME, split=DATASET_SPLIT)
            dataset = apply_chat_template(dataset, tokenizer, CHAT_TEMPLATE)
            dataset = prepare_dataset(dataset, MIN_LEN, MAX_LEN, {"num_proc": 8})
            dataset = cache_dataset(dataset, PREPARED_HASH_PATH)
            READY_FLAG.touch() # mark as finished

    # Everybody arrives here – no need for dist.barrier()
    while not READY_FLAG.exists():
        time.sleep(5)
    
    dataset = load_from_disk(str(PREPARED_HASH_PATH))
    
    batch_sampler = MultipackBatchSampler(
        RandomSampler(dataset),
        lengths=get_dataset_lengths(dataset),
        batch_max_len=MAX_LEN,
        batch_size=TRAIN_MICRO_BATCH_SIZE,
        drop_last=True,
    )

    # NOTE: here we patch model and trainer internals in HF transformers
    patch_for_multipack(batch_sampler)

    collator = V2BatchSamplerDataCollatorForSeq2Seq(
        tokenizer=tokenizer,
        padding="max_length",
        max_length=MAX_LEN,
        pad_to_multiple_of=64,
    )

    trainer = SFTTrainer(
        model=MODEL_PATH,
        processing_class=tokenizer,
        train_dataset=dataset,
        data_collator=collator,
        args=SFTConfig(
            output_dir=OUTPUT_DIR,
            num_train_epochs=1,
            save_strategy="no",
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
                "torch_dtype": "bfloat16",
                "use_cache": False,
            },
            # NOTE: must skip data preparation to work with liger kernels
            use_liger=True,
            dataset_kwargs={
                "skip_prepare_dataset": True,
            },
            deepspeed="./examples/zero3.json"
        ),
    )
    trainer.train()
