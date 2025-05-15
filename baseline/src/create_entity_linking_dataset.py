import json
import argparse
import random

from tqdm import tqdm
from transformers import AutoTokenizer
from typing import List
from src.data_utils import load_data, KGContainer, Example, create_full_entity_description
from src.train_ce_entity_linking_dataset import get_retrieval_elements

INSTRUCT = '''Entity Mention: {}\nEntity Mention Definition: {}\nEntity Mention Types: {}\n\nBased on the above entity mention and its context, identify the ID of the candidate in the following to which the entity mention refers:{}'''

INSTRUCT_WITH_NONE_CASE = '''Entity Mention: {}\nEntity Mention Definition: {}\nEntity Mention Types: {}\n\nBased on the above entity mention and its context, identify the ID of the candidate in the following to which the entity mention refers (if none of them, assign the ID as "None"):{}'''


CANDIDATE_NUM = 10

MAX_INPUT_LENGTH = 4000


def load_json_data(path):
    with open(path, 'r', encoding='utf-8') as fp:
        data = json.load(fp)
    return data


def load_jsonl_data(path):
    data = []
    with open(path, 'r', encoding='utf-8') as fp:
        for line in fp:
            obj = json.loads(line.strip())
            data.append(obj)
    return data


def dump_json_data(data, path):
    with open(path, 'w', encoding='utf-8') as fp:
        json.dump(data, fp, ensure_ascii=True,
                  indent=2, separators=(", ", ": "))


def shorten_entity_description(entity_description, max_len):
    entity_description_tokens = entity_description.split(" ")
    entity_description = ' '.join(entity_description_tokens[: max_len])
    return entity_description


def formulate_candidates(candidate_list, max_len):
    candidates = ""
    candidate_template = '\n\nID: {}\nEntity: {}\nEntity Description: {}\nEntity Types: {}'
    random.shuffle(candidate_list)
    for i, candidate_obj in enumerate(candidate_list):
        entity_description = shorten_entity_description(
            candidate_obj["entity_description"], max_len)
        candidate = candidate_template.format(
            i, candidate_obj["title"], entity_description, ", ".join(candidate_obj["entity_types"]))
        candidates += candidate

    return candidates


def is_length_valid(model_path, human_value, gpt_value, tokenizer):
    messages = [
        {"role": "user", "content": human_value},
        {"role": "assistant", "content": gpt_value}
    ]
    prompt = tokenizer.apply_chat_template(messages, add_generation_prompt=True, tokenize=False)
    # vicuna_convo = get_conversation_template(model_path)
    # vicuna_convo.append_message(vicuna_convo.roles[0], human_value)
    # vicuna_convo.append_message(vicuna_convo.roles[1], gpt_value)
    # prompt = vicuna_convo.get_prompt()

    inputs = tokenizer([prompt])
    input_length = len(inputs["input_ids"][0])

    if random.randint(1, 100) == 1:
        print(f"vicuna input length + output length = {input_length}")

    if input_length > MAX_INPUT_LENGTH:
        return False

    return True


def create_sft_data(context_candidates_list, sft_data_path, model_path, with_none_case=False):
    if with_none_case:
        instruct = INSTRUCT_WITH_NONE_CASE
    else:
        instruct = INSTRUCT
    sft_data = []
    tokenizer = AutoTokenizer.from_pretrained(model_path, use_fast=True)

    shorten_len = 32
    for each in tqdm(context_candidates_list):
        exp = {"mention_id": each["mention_id"],
               "mention": each["mention"]}
        mention_types = [x for x in each["mention_types"] if isinstance(x, str)]

        entity_description_max_len = 256
        keep_shorten = True
        while keep_shorten:
            human_value = instruct.format(
                each["mention"],
                each["mention_definition"],
                ", ".join(mention_types),
                formulate_candidates(
                    each["candidates"], entity_description_max_len)
            )

            ground_truth = {"ID": each["label_id"]}
            gpt_value = json.dumps(ground_truth)

            if is_length_valid(model_path, human_value, gpt_value, tokenizer):
                keep_shorten = False
            else:
                entity_description_max_len -= shorten_len

        messages = [
            {"role": "user", "content": human_value},
            {"role": "assistant", "content": gpt_value}
        ]
        exp["conversations"] = messages
        sft_data.append(exp)

    dump_json_data(sft_data, sft_data_path)



def retrieve_candidates(entities,
                        model, candidate_index, candidate_mapping) -> set:
    prompts = []
    prompt_to_add = ""
    for entity in entities:
        prompt = prompt_to_add + create_full_entity_description(entity.candidate_mention, entity.candidate_definition
                                                                , entity.candidate_types)
        prompts.append(prompt)

    embeddings = model.encode(prompts, show_progress_bar=True, normalize_embeddings=True)
    nearest_indices = candidate_index.search(embeddings, 2 * CANDIDATE_NUM)[1]
    all_candidates = []
    for nearest_indices_ in nearest_indices:
        candidates = set()
        for idx in nearest_indices_:
            if idx in candidate_mapping:
                candidates.add(candidate_mapping[idx]["identifier"])
        all_candidates.append(candidates)
    return all_candidates



def prepare_context_candidates(data: List[Example], entity_index, entity_mapping, candidate_retrieval_model, with_none_case=False):
    model, candidate_index, candidate_mapping = get_retrieval_elements(entity_index, entity_mapping, candidate_retrieval_model)

    all_entities = []
    for item in data:
        for entity in item.entities:
            if entity.candidate_mention is None:
                continue
            all_entities.append(entity)
    all_candidates = retrieve_candidates(
        all_entities,
        model, candidate_index, candidate_mapping)
    none_case_num = 0
    context_candidates_list = []
    counter = 0
    recall = 0
    for entity, candidates in zip(all_entities, all_candidates):
        if entity.candidate_mention is None:
            continue
        candidates = list(candidates)
        if entity.qid in candidates:
            recall += 1
        candidates_list = [candidates[:CANDIDATE_NUM]]
        if random.random() < 0.25:
            if entity.qid in candidates:
                candidates_copy = candidates.copy()
                candidates_copy.remove(entity.qid)
                candidates_list.append(candidates_copy[:CANDIDATE_NUM])
        for candidates in candidates_list:
            random.shuffle(candidates)
            if entity.qid not in candidates:
                if with_none_case:
                    label_id = "None"
                    none_case_num += 1
                else:
                    continue
            else:
                label_id = candidates.index(entity.qid)
            candidate_reps = []
            for candidate_qid in candidates:
                label = kg_container.label(candidate_qid)
                description = kg_container.definition(candidate_qid)
                entity_types = kg_container.types(candidate_qid)
                candidate = {
                    "title": label,
                    "entity_description": description.strip(),
                    "entity_types": entity_types[:3],
                }
                candidate_reps.append(candidate)

            context_candidates = {
                "mention_id": counter,
                "mention": entity.candidate_mention,
                "mention_definition": entity.candidate_definition,
                "mention_types": entity.candidate_types,
                "candidates": candidate_reps,
                "label_id": label_id
            }
            counter += 1
            context_candidates_list.append(context_candidates)

    print(f"num of none case: {none_case_num} out of {counter}")
    print(f"recall: {recall / len(all_entities)} out of {len(all_entities)}")
    return context_candidates_list



if __name__ == "__main__":
    argument_parser = argparse.ArgumentParser()
    argument_parser.add_argument("training_data_path", type=str)

    argument_parser.add_argument("--development_data_path", type=str)
    argument_parser.add_argument("--model_path", type=str, default="meta-llama/Llama-3.1-8B-Instruct")
    argument_parser.add_argument("--with_none_case", action="store_true")
    argument_parser.add_argument("--candidate_retrieval_model", type=str, default="candidate_retriever/final")
    argument_parser.add_argument("--entity_index", type=str, default="entity_index.index")
    argument_parser.add_argument("--entity_mapping", type=str, default="entity_index.json")


    args = argument_parser.parse_args()

    entity_index = args.entity_index
    entity_mapping = args.entity_mapping
    candidate_retrieval_model = args.candidate_retrieval_model

    kg_container = KGContainer()
    train_data = load_data(args.training_data_path, kg_container)
    if args.development_data_path is None:
        assert args.development_data_path is None, "Both dev should be provided or none of them."
        random.shuffle(train_data)
        dev_data = train_data[:int(len(train_data) * 0.2)]
        train_data = train_data[int(len(train_data) * 0.2):]
    else:
        dev_data = load_data(args.development_data_path, kg_container)

    context_candidates_list = prepare_context_candidates(train_data, entity_index, entity_mapping, candidate_retrieval_model, True)
    dev_context_candidates_list = prepare_context_candidates(dev_data, entity_index, entity_mapping, candidate_retrieval_model, True)

    create_sft_data(context_candidates_list,
                    "el_rerank_train", args.model_path, args.with_none_case)

    create_sft_data(dev_context_candidates_list,
                    "el_rerank_dev", args.model_path, args.with_none_case)


