import dataclasses
import json
import multiprocessing
import os
from argparse import ArgumentParser
from typing import List

import torch
from tqdm import tqdm

from vllm import LLM, SamplingParams

from src.data_utils import Example, load_data, KGContainer

import jsonlines


def parse_output(output: str) -> List[dict]:
    start_bracket = output.find("[")
    end_bracket = output.rfind("]")
    if start_bracket != -1 and end_bracket != -1:
        try:
            content = json.loads(output[start_bracket:end_bracket + 1])
        except json.JSONDecodeError:
            return []
        if isinstance(content, list):
            content = [x for x in content if isinstance(x, dict)]
            return content
        else:
            return []
    return []
def generate_definitions(examples: List[Example], kg_container: KGContainer, output_file: str, batch_size: int = 50000):
    model = "meta-llama/Llama-3.1-8B-Instruct"  # Choose any available model
    model = LLM(model=model, max_model_len=4096) # dtype=torch.bfloat16, trust_remote_code=True, quantization="bitsandbytes",
                #load_format="bitsandbytes", ,)

    all_messages = []
    all_entity_labels = []
    main_prompt = open("prompt.txt").read()
    for example in tqdm(examples):
        entity_labels = [kg_container.label(x.qid) for x in example.entities if x.qid in kg_container.entities]
        joined_entities = '\n'.join(entity_labels)
        user_prompt = f"Text: {example.text}\nEntities: \n{joined_entities}\n\n"
        messages = [{"role":"system","content":main_prompt},{"role":"user","content":user_prompt}]

        tokenized_messages = model.llm_engine.tokenizer.tokenizer.apply_chat_template(messages, add_generation_prompt=True, tokenize=False)

        all_messages.append(tokenized_messages)
        all_entity_labels.append(set(entity_labels))
    print(len(all_messages))
    expected_keys = {"entity", "surface_form", "definition", "type"}
    output_file = jsonlines.open(output_file, "w")
    missing_entities = 0
    overall_entities = 0
    for i in range(0, len(all_messages), batch_size):
        batch=  all_messages[i:i + batch_size]
        batch_entity_labels = all_entity_labels[i:i + batch_size]
        batch_examples = examples[i:i + batch_size]
        generations = model.generate(
            batch,
            sampling_params=SamplingParams(
                max_tokens=512, n=1
            ),
            use_tqdm=True
        )

        for gen, entity_labels, example in tqdm(zip(generations, batch_entity_labels, batch_examples)):
            generated_entities = []
            encountered = set()
            outputs = []
            for output in gen.outputs:
                choice = output.text
                outputs.append(choice)
                parsed = parse_output(choice)

                for x in parsed:
                    if len(expected_keys.intersection(x.keys())) != len(expected_keys):
                        continue
                    try:
                        if x["entity"] in entity_labels and x["entity"] not in encountered:
                            x = {key.strip(): value for key, value in x.items()}
                            encountered.add(x["entity"])
                            generated_entities.append(x)
                    except:
                        continue
            overall_entities += len(entity_labels)
            missing_entities += len(entity_labels) - len(encountered)

            output_file.write({
                **dataclasses.asdict(example),
                "generated_text": outputs,
                "generated_entities": generated_entities,
            })

        print(f"Ratio of missing entities: {missing_entities / overall_entities}")

def generate_worker(gpu_id: int, batch_examples: List[Example], kg_container: KGContainer, output_file: str, n: int):
    os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu_id)
    model = LLM(model="meta-llama/Llama-3.1-8B-Instruct", max_model_len=4096)

    all_messages = []
    all_entity_labels = []
    main_prompt = open("prompt.txt").read()

    for example in tqdm(batch_examples, desc=f"GPU {gpu_id} prompt prep"):
        entity_labels = [kg_container.label(x.qid) for x in example.entities if x.qid in kg_container.entities]
        joined_entities = '\n'.join(entity_labels)
        user_prompt = f"Text: {example.text}\nEntities: \n{joined_entities}\n\n"
        messages = [{"role": "system", "content": main_prompt},
                    {"role": "user", "content": user_prompt}]
        tokenized = model.llm_engine.tokenizer.tokenizer.apply_chat_template(messages, add_generation_prompt=True, tokenize=False)

        all_messages.append(tokenized)
        all_entity_labels.append(set(entity_labels))

    output_path = f"{output_file.replace('.jsonl', '')}_gpu{gpu_id}.jsonl"
    output_file = jsonlines.open(output_path, "w")

    expected_keys = {"entity", "surface_form", "definition", "type"}
    missing_entities = 0
    overall_entities = 0
    batch_size: int = 10000

    for i in tqdm(range(0, len(all_messages), batch_size), desc=f"GPU {gpu_id} generation"):
        batch = all_messages[i:i + batch_size]
        batch_labels = all_entity_labels[i:i + batch_size]
        batch_exs = batch_examples[i:i + batch_size]

        generations = model.generate(batch, sampling_params=SamplingParams(max_tokens=512, n=n), use_tqdm=False)

        for gen, entity_labels, example in zip(generations, batch_labels, batch_exs):
            generated_entities = []
            encountered = set()
            outputs = []

            for output in gen.outputs:
                choice = output.text
                outputs.append(choice)
                parsed = parse_output(choice)

                for x in parsed:
                    if len(expected_keys.intersection(x.keys())) != len(expected_keys):
                        continue
                    try:
                        if x["entity"] in entity_labels and x["entity"] not in encountered:
                            x = {key.strip(): value for key, value in x.items()}
                            encountered.add(x["entity"])
                            generated_entities.append(x)
                    except:
                        continue

            overall_entities += len(entity_labels)
            missing_entities += len(entity_labels) - len(encountered)

            output_file.write({
                **dataclasses.asdict(example),
                "generated_text": outputs,
                "generated_entities": generated_entities,
            })

    print(f"[GPU {gpu_id}] Ratio of missing entities: {missing_entities / overall_entities:.2%}")



if __name__ == "__main__":
    # file_name = "data.json"
    argparser = ArgumentParser()
    argparser.add_argument("input_file", type=str)
    argparser.add_argument("output_file", type=str)
    argparser.add_argument("-n", type=int, default=1)
    kg_container = KGContainer()
    args = argparser.parse_args()
    examples = load_data(args.input_file, kg_container)
    # generate_definitions(examples, kg_container, args.output_file)

    # Automatically detect number of GPUs
    num_gpus = torch.cuda.device_count()
    print(f"Detected {num_gpus} GPU(s). Spawning one process per GPU.")

    # Divide examples evenly across GPUs
    chunks = [examples[i::num_gpus] for i in range(num_gpus)]

    processes = []
    for i in range(num_gpus):
        p = multiprocessing.Process(
            target=generate_worker,
            args=(i, chunks[i], kg_container, args.output_file, args.n)
        )
        p.start()
        processes.append(p)

    for p in processes:
        p.join()

    print("✅ All GPU processes completed.")

