import json
import os

import torch
from accelerate import Accelerator
from peft import LoraConfig, TaskType, get_peft_model
from transformers import AutoTokenizer, AutoModelForCausalLM, TrainingArguments, DataCollatorForLanguageModeling, \
    Trainer
from datasets import Dataset



def train(
    dataset_path: str,
    dev_dataset_path: str,
    model_name: str = "meta-llama/Llama-3.1-8B-Instruct",
    output_dir: str = "./llama3-lora-finetune",
    save_dir: str = "./llama3-lora-entity-linking",
    max_length: int = 512,
    per_device_train_batch_size: int = 4,
    gradient_accumulation_steps: int = 4,
    lr: float = 2e-5,
    epochs: int = 1,
    lora_r: int = 16,
    lora_alpha: int = 32,
    lora_dropout: float = 0.1,
):
    accelerator = Accelerator()
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    tokenizer.pad_token = tokenizer.eos_token  # Make sure padding works

    if not os.path.exists(dataset_path + ".pkl"):
        def convert(example):
            return {"text": tokenizer.apply_chat_template(example["conversations"], add_generation_prompt=True, tokenize=False)}

        data = json.load((open(dataset_path, "r")))
        print(f"Number of examples in the dataset: {len(data)}")
        data = [x for x in data if isinstance(x["mention"], str)]
        print(f"Number of examples in the dataset after filtering: {len(data)}")
        dataset = Dataset.from_list(data).map(convert, )

        dev_data = json.load((open(dev_dataset_path, "r")))
        print(f"Number of examples in the dev dataset: {len(dev_data)}")
        dev_data = [x for x in dev_data if isinstance(x["mention"], str)]
        print(f"Number of examples in the dev dataset after filtering: {len(dev_data)}")
        dev_dataset = Dataset.from_list(dev_data).map(convert, )

        def tokenize(example):
            return tokenizer(example["text"], truncation=True, padding="max_length", max_length=max_length)

        tokenized_dataset = dataset.map(tokenize, )
        tokenized_dev_dataset = dev_dataset.map(tokenize, )

        tokenized_dataset.save_to_disk(dataset_path + ".pkl")
        tokenized_dev_dataset.save_to_disk(dev_dataset_path + ".pkl")
    else:
        tokenized_dataset = Dataset.load_from_disk(dataset_path + ".pkl")
        tokenized_dev_dataset = Dataset.load_from_disk(dev_dataset_path + ".pkl")


    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        device_map=None,  # ⛔ disables auto-sharding (causes DTensor issues)
        torch_dtype=torch.float16,
        attn_implementation="flash_attention_2",
        low_cpu_mem_usage=False,  # ensures no sharded loading
    )

    lora_config = LoraConfig(
        r=lora_r,
        lora_alpha=lora_alpha,
        target_modules=["q_proj", "v_proj"],
        lora_dropout=lora_dropout,
        bias="none",
        task_type=TaskType.CAUSAL_LM
    )

    model = get_peft_model(model, lora_config)
    effective_batch_size = per_device_train_batch_size * gradient_accumulation_steps
    total_steps = len(tokenized_dataset) // effective_batch_size

    training_args = TrainingArguments(
        output_dir=output_dir,
        per_device_train_batch_size=per_device_train_batch_size,
        gradient_accumulation_steps=gradient_accumulation_steps,
        learning_rate=lr,
        logging_steps=5,
        eval_steps=total_steps // 100,
        eval_strategy="steps",
        save_steps=total_steps // 100,
        save_total_limit=2,
        load_best_model_at_end=True,
        save_strategy="steps",
        num_train_epochs=epochs,
        fp16=True,
        optim="paged_adamw_8bit",
    )

    data_collator = DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=False)

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=tokenized_dataset,
        eval_dataset=tokenized_dev_dataset,
        data_collator=data_collator
    )

    trainer.train()

    model.save_pretrained(save_dir)
    tokenizer.save_pretrained(save_dir)

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Fine-tune LLM with LoRA")
    parser.add_argument("--dataset_path", type=str, default="el_rerank_train", help="Path to the training dataset")
    parser.add_argument("--dev_dataset_path", type=str, default="el_rerank_dev", help="Path to the dev dataset")
    parser.add_argument("--model_name", type=str, default="meta-llama/Llama-3.1-8B-Instruct", help="Model name")
    parser.add_argument("--output_dir", type=str, default="./llama3-lora-finetune", help="Training checkpoint directory")
    parser.add_argument("--save_dir", type=str, default="./llama3-lora-entity-linking", help="Final model save directory")
    parser.add_argument("--max_length", type=int, default=512, help="Max token length for tokenization")
    parser.add_argument("--per_device_train_batch_size", type=int, default=4, help="Per-device training batch size")
    parser.add_argument("--gradient_accumulation_steps", type=int, default=4, help="Gradient accumulation steps")
    parser.add_argument("--lr", type=float, default=2e-5, help="Learning rate")
    parser.add_argument("--epochs", type=int, default=1, help="Number of training epochs")
    parser.add_argument("--lora_r", type=int, default=16, help="LoRA rank")
    parser.add_argument("--lora_alpha", type=int, default=32, help="LoRA alpha")
    parser.add_argument("--lora_dropout", type=float, default=0.1, help="LoRA dropout")
    args = parser.parse_args()

    train(
        args.dataset_path,
        args.dev_dataset_path,
        model_name=args.model_name,
        output_dir=args.output_dir,
        save_dir=args.save_dir,
        max_length=args.max_length,
        per_device_train_batch_size=args.per_device_train_batch_size,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        lr=args.lr,
        epochs=args.epochs,
        lora_r=args.lora_r,
        lora_alpha=args.lora_alpha,
        lora_dropout=args.lora_dropout,
    )