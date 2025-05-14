import random
from argparse import ArgumentParser
from typing import List
from datasets import Dataset,DatasetDict
from src.data_utils import load_data, KGContainer, Example
from src.special_prompts import RE_RET_PROMPT


def create_dataset(data: List[Example], kg_container: KGContainer, add_special_prompt: bool, num_negatives: int = 1) -> Dataset:
    dataset = []
    prompt_to_add = RE_RET_PROMPT if add_special_prompt else ""
    relation_idx = {}

    for item in data:
        existing_relations = {p for s, p, o in item.triples}
        negative_relations = random.sample(kg_container.relations.keys() - existing_relations,
                                           len(existing_relations) * num_negatives)
        for relation in existing_relations:
            if relation not in relation_idx:
                relation_idx[relation] = len(relation_idx)

        for relation in negative_relations:
            if relation not in relation_idx:
                relation_idx[relation] = len(relation_idx)
        for idx, relation in enumerate(existing_relations):
            selected_negative_relations = negative_relations[idx * num_negatives:(idx + 1) * num_negatives]
            for negative_relation in selected_negative_relations:
                try:
                    dataset.append({
                        "sentence": prompt_to_add + item.text,
                        "positive": f"{kg_container.label(relation)}: {kg_container.definition(relation)}",
                        "negative": f"{kg_container.label(negative_relation)}: {kg_container.definition(negative_relation)}",
                        "all_positive_relations": [relation_idx[r] for r in existing_relations],
                        "positive_relation": relation_idx[relation],
                        "negative_relation": relation_idx[negative_relation],
                    })
                except KeyError:
                    print(f"KeyError for relation {relation}")
                    continue


    return Dataset.from_list(dataset)


if __name__ == "__main__":
    argument_parser = ArgumentParser()
    argument_parser.add_argument("training_data_path", type=str)
    argument_parser.add_argument("--development_data_path", type=str)
    argument_parser.add_argument("--add_special_prompt", action="store_true", help="Add special prompt to the sentence")
    args = argument_parser.parse_args()


    kg_container = KGContainer()

    train_data = load_data(args.training_data_path, kg_container)
    if args.development_data_path is None:
        assert args.development_data_path is None , "Both dev should be provided or none of them."
        random.shuffle(train_data)
        dev_data = train_data[:int(len(train_data) * 0.2)]
        train_data = train_data[int(len(train_data) * 0.2):]
    else:
        dev_data = load_data(args.development_data_path, kg_container)

    train_test_valid_dataset = DatasetDict({
        'train': create_dataset(train_data, kg_container, args.add_special_prompt),
        'valid': create_dataset(dev_data, kg_container, args.add_special_prompt),})

    train_test_valid_dataset.save_to_disk("schema_gen_dataset/")