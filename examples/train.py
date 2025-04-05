from trl import SFTTrainer, SFTConfig
from datasets import load_dataset, Dataset
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


if __name__ == "__main__":
    tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH)
    tokenizer.pad_token = tokenizer.eos_token

    dataset = load_dataset(DATASET_PATH, DATASET_NAME, split=DATASET_SPLIT)
    dataset = apply_chat_template(dataset, tokenizer, qwen25_template)
    dataset = prepare_dataset(dataset, MIN_LEN, MAX_LEN, {"num_proc": 8})

    batch_sampler = MultipackBatchSampler(
        RandomSampler(dataset),
        lengths=get_dataset_lengths(dataset),
        packing_efficiency_estimate=1.0,
        batch_max_len=TRAIN_MICRO_BATCH_SIZE * MAX_LEN,
        batch_size=1,
        drop_last=True,
    )

    # NOTE: here we patch model and trainer internals in HF transformers
    patch_for_multipack(batch_sampler)

    collator = V2BatchSamplerDataCollatorForSeq2Seq(
        tokenizer=tokenizer,
        padding=True,
    )

    trainer = SFTTrainer(
        model=MODEL_PATH,
        processing_class=tokenizer,
        train_dataset=dataset,
        data_collator=collator,
        args=SFTConfig(
            max_steps=10,
            output_dir=OUTPUT_DIR,
            dataset_text_field="text",
            max_seq_length=MAX_LEN,
            dataset_num_proc=8,
            per_device_train_batch_size=TRAIN_MICRO_BATCH_SIZE,
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
            model_init_kwargs={"attn_implementation": "flash_attention_2"},
        )
    )
    trainer.train()
