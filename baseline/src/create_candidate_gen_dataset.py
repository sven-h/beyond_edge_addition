import random
from argparse import ArgumentParser
from typing import List
from datasets import Dataset,DatasetDict
from tqdm import tqdm

from src.data_utils import load_data, KGContainer, Example, create_full_entity_description
from src.special_prompts import STS_PROMPT


def create_dataset(data: List[Example], kg_container: KGContainer, add_special_prompt: bool, num_negatives: int = 1) -> Dataset:
    dataset = []
    prompt_to_add = STS_PROMPT if add_special_prompt else ""

    all_entity_qids = list(kg_container.entity_qids)

    for item in tqdm(data):
        gt_descriptions = {
            create_full_entity_description(kg_container.label(entity.qid), kg_container.definition(entity.qid), kg_container.types(entity.qid)) for entity in
            item.entities}
        existing_qids = {entity.qid for entity in item.entities}
        negative_entities = random.sample(all_entity_qids,
                                           len(existing_qids) * num_negatives * 2)
        negative_entities = [qid for qid in negative_entities if qid not in existing_qids][:len(existing_qids) * num_negatives]
        for idx, entity in enumerate(item.entities):
            if not entity.candidate_mention or not entity.candidate_definition:
                continue
            selected_negative_entities= negative_entities[idx * num_negatives:(idx + 1) * num_negatives]
            selected_negative_descriptions = {create_full_entity_description(kg_container.label(qid), kg_container.definition(qid),
                                                                                   kg_container.types(qid)) for qid in selected_negative_entities}
            candidate_description = create_full_entity_description(entity.candidate_mention,
                                                                   entity.candidate_definition,
                                                                   entity.candidate_types)
            positive_description = create_full_entity_description(kg_container.label(entity.qid),
                                                                   kg_container.definition(entity.qid),
                                                                   kg_container.types(entity.qid))
            # all_negatives = (gt_descriptions - {positive_description}) | selected_negative_descriptions
            for negative_description in selected_negative_descriptions :
                dataset.append({
                    "sentence": prompt_to_add + candidate_description,
                    "positive": positive_description,
                    "negative": negative_description
                })

    return Dataset.from_list(dataset)


if __name__ == "__main__":
    argument_parser = ArgumentParser()
    argument_parser.add_argument("training_data_path", type=str)
    argument_parser.add_argument("--development_data_path", type=str)
    argument_parser.add_argument("--add_special_prompt", action="store_true", help="Add special prompt to the sentence")
    argument_parser.add_argument("--test_data_path", type=str)
    argument_parser.add_argument("--output_path", type=str, default="candidate_gen_dataset/",)
    args = argument_parser.parse_args()

    kg_container = KGContainer()
    train_data = load_data(args.training_data_path, kg_container)
    if args.development_data_path is None:
        assert args.development_data_path is None, "Both dev should be provided or none of them."
        random.shuffle(train_data)
        dev_data = train_data[:int(len(train_data) * 0.2)]
        train_data = train_data[int(len(train_data) * 0.2):]
    else:
        dev_data = load_data(args.development_data_path, kg_container)

    train_test_valid_dataset = DatasetDict({
        'train': create_dataset(train_data, kg_container, args.add_special_prompt),
        'valid': create_dataset(dev_data, kg_container, args.add_special_prompt),})

    train_test_valid_dataset.save_to_disk(args.output_path)