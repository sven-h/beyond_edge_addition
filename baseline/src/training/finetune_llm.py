import json
import os

import torch
from accelerate import Accelerator
from peft import LoraConfig, TaskType, get_peft_model
from transformers import AutoTokenizer, AutoModelForCausalLM, TrainingArguments, DataCollatorForLanguageModeling, \
    Trainer
from datasets import Dataset



def train(dataset_path: str, dev_dataset_path: str, model_name: str=  "meta-llama/Llama-3.1-8B-Instruct"):
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
            return tokenizer(example["text"], truncation=True, padding="max_length", max_length=512)

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
        r=16,
        lora_alpha=32,
        target_modules=["q_proj", "v_proj"],
        lora_dropout=0.1,
        bias="none",
        task_type=TaskType.CAUSAL_LM
    )

    model = get_peft_model(model, lora_config)
    batch_size = 8
    epochs = 1
    total_steps = len(tokenized_dataset) // batch_size

    training_args = TrainingArguments(
        output_dir="./llama3-lora-finetune",
        per_device_train_batch_size=4,
        gradient_accumulation_steps=4,
        learning_rate=2e-5,
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

    model.save_pretrained("./llama3-lora-entity-linking")
    tokenizer.save_pretrained("./llama3-lora-entity-linking")

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Fine-tune LLM with LoRA")
    parser.add_argument("--dataset_path", type=str, default="el_rerank_train", help="Path to the dataset")
    parser.add_argument("--dev_dataset_path", type=str, default="el_rerank_dev", help="Path to the dataset")
    parser.add_argument("--model_name", type=str, default="meta-llama/Llama-3.1-8B-Instruct", help="Model name")
    args = parser.parse_args()

    train(args.dataset_path, args.dev_dataset_path, args.model_name)