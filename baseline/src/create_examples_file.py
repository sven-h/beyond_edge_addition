import argparse
import json
from collections import defaultdict

from src.data_utils import KGContainer, load_data


def main():
    argparser = argparse.ArgumentParser(description="Create relation schema CSV.")
    argparser.add_argument("--kg_data_path", type=str, help="Path to the knowledge graph data.", default="data")
    args = argparser.parse_args()
    kg_container = KGContainer(args.kg_data_path)

    path = "train_dataset.jsonl"

    train_data = load_data(path, kg_container)


    example_dict = defaultdict(list)
    for item in train_data:
        for s, p, o in item.triples:
            if s not in kg_container.entities or p not in kg_container.relations or o not in kg_container.entities:
                continue
            s_label = kg_container.label(s)
            o_label = kg_container.label(o)
            p_label = kg_container.label(p)

            example_dict[p_label].append({
                "text": item.text,
                "triplet": [s_label, p_label, o_label],
            })
    json.dump(example_dict, open("examples.json", "w"), indent=4)


if __name__ == "__main__":
    main()