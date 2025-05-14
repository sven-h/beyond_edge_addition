import multiprocessing
import os

import numpy as np
import openai
import time

from sklearn.preprocessing import normalize
from transformers import AutoModelForCausalLM, AutoTokenizer, GenerationConfig
import ast
from sentence_transformers import SentenceTransformer
from typing import List, Union
import gc
import torch
import logging

from vllm import SamplingParams, LLM
from vllm.lora.request import LoRARequest

logger = logging.getLogger(__name__)


def free_model(model: AutoModelForCausalLM = None, tokenizer: AutoTokenizer = None):
    try:
        model.cpu()
        if model is not None:
            del model
        if tokenizer is not None:
            del tokenizer
        gc.collect()
        torch.cuda.empty_cache()
    except Exception as e:
        logger.warning(e)


def get_embedding_e5mistral(model, tokenizer, sentence, task=None):
    model.eval()
    device = model.device

    if task != None:
        # It's a query to be embed
        sentence = get_detailed_instruct(task, sentence)

    sentence = [sentence]

    max_length = 4096
    # Tokenize the input texts
    batch_dict = tokenizer(
        sentence, max_length=max_length - 1, return_attention_mask=False, padding=False, truncation=True
    )
    # append eos_token_id to every input_ids
    batch_dict["input_ids"] = [input_ids + [tokenizer.eos_token_id] for input_ids in batch_dict["input_ids"]]
    batch_dict = tokenizer.pad(batch_dict, padding=True, return_attention_mask=True, return_tensors="pt")

    batch_dict.to(device)

    embeddings = model(**batch_dict).detach().cpu()

    assert len(embeddings) == 1

    return embeddings[0]


def get_detailed_instruct(task_description: str, query: str) -> str:
    return f"Instruct: {task_description}\nQuery: {query}"


def get_embedding_sts(model: SentenceTransformer, texts: List[str], prompt_name=None, prompt=None):
    adapter_path = None
    if isinstance(model, tuple):
        model, adapter_path = model

    if isinstance(model, LLM):
        prompts = []
        for text in texts:
            if prompt is not None:
                current_prompt = prompt + text
            else:
                current_prompt = text
            prompts.append(current_prompt)
        if adapter_path:
            embeddings = model.embed(prompts, lora_request=LoRARequest(adapter_path, hash(adapter_path), adapter_path))
        else:
            embeddings = model.embed(prompts)
        embeddings = np.array([x.outputs.embedding for x  in embeddings], dtype=np.float32)
        embeddings = normalize(embeddings, axis=1)
    else:
        embeddings = model.encode(texts, prompt_name=prompt_name, prompt=prompt, normalize_embeddings=True)
    return embeddings


def parse_raw_entities(raw_entities: str):
    parsed_entities = []
    try:
        left_bracket_idx = raw_entities.index("[")
        right_bracket_idx = raw_entities.index("]")
        parsed_entities = ast.literal_eval(raw_entities[left_bracket_idx : right_bracket_idx + 1])
    except Exception as e:
        pass
    logging.debug(f"Entities {raw_entities} parsed as {parsed_entities}")
    return parsed_entities


def parse_raw_triplets(raw_triplets: str):
    # Look for enclosing brackets
    unmatched_left_bracket_indices = []
    matched_bracket_pairs = []

    collected_triples = []
    for c_idx, c in enumerate(raw_triplets):
        if c == "[":
            unmatched_left_bracket_indices.append(c_idx)
        if c == "]":
            if len(unmatched_left_bracket_indices) == 0:
                continue
            # Found a right bracket, match to the last found left bracket
            matched_left_bracket_idx = unmatched_left_bracket_indices.pop()
            matched_bracket_pairs.append((matched_left_bracket_idx, c_idx))
    for l, r in matched_bracket_pairs:
        bracketed_str = raw_triplets[l : r + 1]
        try:
            parsed_triple = ast.literal_eval(bracketed_str)
            if len(parsed_triple) == 3 and all([isinstance(t, str) for t in parsed_triple]):
                if all([e != "" and e != "_" for e in parsed_triple]):
                    collected_triples.append(parsed_triple)
            elif not all([type(x) == type(parsed_triple[0]) for x in parsed_triple]):
                for e_idx, e in enumerate(parsed_triple):
                    if isinstance(e, list):
                        parsed_triple[e_idx] = ", ".join(e)
                collected_triples.append(parsed_triple)
        except Exception as e:
            pass
    logger.debug(f"Triplets {raw_triplets} parsed as {collected_triples}")
    return collected_triples


def parse_relation_definition(raw_definitions: str):
    descriptions = raw_definitions.split("\n")
    relation_definition_dict = {}

    for description in descriptions:
        if ":" not in description:
            continue
        index_of_colon = description.index(":")
        relation = description[:index_of_colon].strip()
        if relation.startswith("-"):
            relation = relation[1:].strip()

        relation_description = description[index_of_colon + 1 :].strip()


        if relation == "Answer":
            continue

        relation_definition_dict[relation] = relation_description
    logger.debug(f"Relation Definitions {raw_definitions} parsed as {relation_definition_dict}")
    return relation_definition_dict


def is_model_openai(model_name):
    return "gpt" in model_name


def generate_worker(idx: int, model: LLM, batch_examples: list, sampling_params: SamplingParams, deactivate_pbar: bool, queue: multiprocessing.Queue):
    generation = model.generate(batch_examples, sampling_params=sampling_params, use_tqdm=not deactivate_pbar)
    queue.put((generation, idx))
def generate_completion_transformers(
    input_list: list,
    model: Union[LLM, List[LLM]],
    tokenizer: AutoTokenizer,
    max_new_token=1024,
    answer_prepend="",
    deactivate_pbar = False
):
    tokenizer.pad_token = tokenizer.eos_token

    all_messages = []
    for input in input_list:
        messages = tokenizer.apply_chat_template(input, add_generation_prompt=True, tokenize=False) + answer_prepend
        all_messages.append(messages)

    # model_inputs = tokenizer(messages, return_tensors="pt", padding=True, add_special_tokens=False).to(device)

    generation_config = GenerationConfig(
        do_sample=False,
        max_new_tokens=max_new_token,
        pad_token_id=tokenizer.eos_token_id,
        return_dict_in_generate=True,
    )

    sampling_params = SamplingParams(
        max_tokens=max_new_token,
        temperature=0.0,
    )

    if isinstance(model, list):
        # Split data and run parallel
        num_models = len(model)
        num_inputs = len(all_messages)
        chunk_size = num_inputs // num_models
        if num_inputs % num_models != 0:
            chunk_size += 1
        all_messages = [all_messages[i : i + chunk_size] for i in range(0, num_inputs, chunk_size)]
        processes = []
        queue = multiprocessing.Queue()
        for i in range(num_models):
            p = multiprocessing.Process(
                target=generate_worker,
                args=(i, model[i], all_messages[i], sampling_params, deactivate_pbar, queue),
            )
            processes.append(p)
            p.start()
        for p in processes:
            p.join()
        idx_to_generation = {}
        for i in range(len(processes)):
            gen, idx = queue.get()
            idx_to_generation[idx] = gen
        generation = []
        for i in range(len(processes)):
            generation.extend(idx_to_generation[i])
    else:
        generation = model.generate(all_messages, sampling_params=sampling_params, use_tqdm=not deactivate_pbar)
    generated_texts = [x.outputs[0].text.strip() for x in generation]

    return generated_texts


def openai_chat_completion(model, system_prompt, history, temperature=0, max_tokens=512):
    openai.api_key = os.environ["OPENAI_KEY"]
    response = None
    if system_prompt is not None:
        messages = [{"role": "system", "content": system_prompt}] + history
    else:
        messages = history
    while response is None:
        try:
            response = openai.chat.completions.create(
                model=model, messages=messages, temperature=temperature, max_tokens=max_tokens
            )
        except Exception as e:
            time.sleep(5)
    logging.debug(f"Model: {model}\nPrompt:\n {messages}\n Result: {response.choices[0].message.content}")
    return response.choices[0].message.content
