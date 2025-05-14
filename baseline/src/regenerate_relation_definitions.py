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


def parse_output(output: str) -> dict:
    start_bracket = output.find("{")
    end_bracket = output.rfind("}")
    if start_bracket != -1 and end_bracket != -1:
        try:
            content = json.loads(output[start_bracket:end_bracket + 1])
        except json.JSONDecodeError:
            return {}
        if isinstance(content, dict):
            return content
        else:
            return {}
    return {}
def generate_definitions(relations, batch_size: int = 50000):
    model = "meta-llama/Llama-3.1-8B-Instruct"  # Choose any available model
    model = LLM(model=model, max_model_len=4096) # dtype=torch.bfloat16, trust_remote_code=True, quantization="bitsandbytes",
                # load_format="bitsandbytes", ,)

    all_messages = []
    all_relations = []
    main_prompt = open("data/re_prompt.txt").read()
    for relation in tqdm(relations):
        user_prompt = f"{relation.label}: {relation.definition}"
        messages = [{"role":"system","content":main_prompt},{"role":"user","content":user_prompt}]

        tokenized_messages = model.llm_engine.tokenizer.tokenizer.apply_chat_template(messages, add_generation_prompt=True, tokenize=False)
        all_relations.append(relation)

        all_messages.append(tokenized_messages)
    not_regen = 0
    count  = 0
    relation_to_new_description = {}
    for i in range(0, len(all_messages), batch_size):
        batch=  all_messages[i:i + batch_size]
        batch_relations = all_relations[i:i + batch_size]
        generations = model.generate(
            batch,
            sampling_params=SamplingParams(
                max_tokens=512, n=3
            ),
            use_tqdm=True
        )

        for gen, relation in tqdm(zip(generations, batch_relations)):
            outputs = []
            description = None
            for output in gen.outputs:
                choice = output.text
                outputs.append(choice)
                parsed = parse_output(choice)

                if "definition" in parsed and isinstance(parsed.get("definition"), str):
                    description = parsed["definition"]
                relation_to_new_description[relation.label] = description
                if description is not None:
                    break
            if description is None:
                not_regen += 1
            count  += 1

    print(f"Number of relations not regenerated: {not_regen/count}")
    json.dump(
        relation_to_new_description,
        open("data/relation_definitions.json", "w"),
        ensure_ascii=False,
        indent=4,
    )




if __name__ == "__main__":
    kg_container = KGContainer()
    relations = kg_container.relations.values()
    generate_definitions(relations)


