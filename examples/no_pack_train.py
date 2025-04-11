from trl import SFTTrainer, SFTConfig
from datasets import load_dataset, Dataset
from transformers import AutoTokenizer, PreTrainedTokenizer
from flashpack import (
    mistral_template,
)

OUTPUT_DIR = "./outputs"
DATASET_PREPARED_PATH = "./prepared_datasets"
DATASET_PATH = "Yukang/LongAlpaca-12k"
DATASET_NAME = None
DATASET_SPLIT = "train"
CHAT_TEMPLATE = mistral_template
TRAIN_MICRO_BATCH_SIZE = 2
MODEL_PATH = "mistralai/Mistral-Nemo-Base-2407"
MAX_LEN = 65536


def apply_chat_template(
    dataset: Dataset,
    tokenizer: PreTrainedTokenizer,
    chat_template: str,
) -> Dataset:
    def map_fn(example):
        conversation = [
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "user", "content": example["instruction"]},
            {"role": "assistant", "content": example["output"]},
        ]
        formatted_chat = tokenizer.apply_chat_template(
            conversation,
            chat_template=chat_template,
            tokenize=False,
            add_generation_prompt=False,
        )

        return {"text": formatted_chat}

    dataset = dataset.map(
        map_fn,
        num_proc=8,
        desc="Applying Chat Template",
    )

    tokenizer.chat_template = chat_template

    return dataset


if __name__ == "__main__":
    tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH)
    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "left"

    dataset = load_dataset(DATASET_PATH, DATASET_NAME, split=DATASET_SPLIT)
    dataset = apply_chat_template(dataset, tokenizer, CHAT_TEMPLATE)

    trainer = SFTTrainer(
        model=MODEL_PATH,
        processing_class=tokenizer,
        train_dataset=dataset,
        args=SFTConfig(
            output_dir=OUTPUT_DIR,
            num_train_epochs=1,
            save_strategy="no",
            dataset_text_field="text",
            max_seq_length=MAX_LEN,
            dataset_num_proc=8,
            per_device_train_batch_size=TRAIN_MICRO_BATCH_SIZE,
            include_tokens_per_second=True,
            gradient_accumulation_steps=1,
            learning_rate=2e-5,
            max_grad_norm=1.0,
            warmup_ratio=0.1,
            weight_decay=0.01,
            logging_steps=10,
            bf16=True,
            optim="adamw_torch_fused",
            use_liger=True,
            gradient_checkpointing=True,
            gradient_checkpointing_kwargs={"use_reentrant": False},
            model_init_kwargs={
                "attn_implementation": "flash_attention_2",
                "torch_dtype": "bfloat16",
                "use_cache": False,
            },
            deepspeed="./examples/zero3.json"
        ),
    )
    trainer.train()