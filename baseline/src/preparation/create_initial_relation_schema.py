import argparse
import csv

from src.data_utils import KGContainer


def create_relation_schema_csv(kg_container: KGContainer, output_path: str):
    writer = csv.writer(open(output_path, "w"))
    for relation in kg_container.relations:
        writer.writerow([relation, kg_container.label(relation), kg_container.definition(relation)])


if __name__ == "__main__":
    argparser = argparse.ArgumentParser(description="Create relation schema CSV.")
    argparser.add_argument("--kg_data_path", type=str, help="Path to the knowledge graph data.", default="data")
    args = argparser.parse_args()
    kg_container = KGContainer(args.kg_data_path)

    create_relation_schema_csv(kg_container, "relation_schema.csv")