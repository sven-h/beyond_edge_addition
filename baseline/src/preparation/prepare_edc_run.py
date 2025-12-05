import argparse
import os

import jsonlines

from src.preparation.create_initial_relation_schema import create_relation_schema_csv
from src.preparation.create_test_txt import generate_text_file
from src.data_utils import KGContainer
from src.preparation.regenerate_relation_definitions import generate_definitions

if __name__ == "__main__":
    argparser = argparse.ArgumentParser(description="Create relation schema CSV.")
    argparser.add_argument("--kg_data_path", type=str, help="Path to the knowledge graph data.", default="data")
    argparser.add_argument("--prompts_path", type=str, help="Path prompt data.", default="data")
    argparser.add_argument(
        "--test_data_input_path",
        type=str,
        default="test_dataset.jsonl",
        help="Path to the input JSONL file.")
    args = argparser.parse_args()
    kg_container = KGContainer(args.kg_data_path)

    relations = kg_container.relations.values()

    generate_text_file(list(jsonlines.open(args.test_data_input_path, "r")), "test.txt")
    generate_definitions(relations, os.path.join(args.prompts_path, "re_prompt.txt"), os.path.join(args.kg_data_path, "relation_definitions.json"))
    create_relation_schema_csv(kg_container, os.path.join(args.kg_data_path, "relation_schema.csv"))