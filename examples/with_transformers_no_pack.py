from datasets import load_dataset, Dataset
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    PreTrainedTokenizer,
    Trainer,
    TrainingArguments,
)
from flashpack import qwen25_template

OUTPUT_DIR = "./outputs"
DATASET_PREPARED_PATH = "./prepared_datasets"
DATASET_PATH = "HuggingFaceTB/smoltalk"
DATASET_NAME = "everyday-conversations"
DATASET_SPLIT = "train"
DATASET_COLUMN = "messages"
CHAT_TEMPLATE = qwen25_template
TRAIN_MICRO_BATCH_SIZE = 1 # tuned for H100
MODEL_PATH = "Qwen/Qwen2.5-7B-Instruct"
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
            add_generation_prompt=False,
            truncation=True,
            padding="max_length",
            max_length=MAX_LEN,
            return_dict=True,
        )
        formatted_chat["labels"] = formatted_chat["input_ids"].copy()

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

    dataset = load_dataset(DATASET_PATH, DATASET_NAME, split=DATASET_SPLIT)
    dataset = apply_chat_template(dataset, tokenizer, CHAT_TEMPLATE)

    def model_init():
        return AutoModelForCausalLM.from_pretrained(
            MODEL_PATH,
            **{
                "attn_implementation": "flash_attention_2",
                "torch_dtype": "bfloat16",
                "use_cache": False,
            }
        )

    trainer = Trainer(
        model_init=model_init,
        processing_class=tokenizer,
        train_dataset=dataset,
        args=TrainingArguments(
            output_dir=OUTPUT_DIR,
            num_train_epochs=1,
            save_strategy="no",
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
            use_liger_kernel=True,
            # gradient_checkpointing=True,
            # gradient_checkpointing_kwargs={"use_reentrant": False},
            deepspeed="./examples/zero2.json"
        )
    )

    trainer.train()