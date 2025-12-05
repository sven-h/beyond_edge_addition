import argparse
import csv

from src.data_utils import KGContainer

if __name__ == "__main__":
    argparser = argparse.ArgumentParser(description="Create relation schema CSV.")
    argparser.add_argument("--kg_data_path", type=str, help="Path to the knowledge graph data.", default="data")
    args = argparser.parse_args()
    kg_container = KGContainer(args.kg_data_path)
    writer = csv.writer(open("relation_schema.csv", "w"))
    for relation in kg_container.relations:
        writer.writerow([relation, kg_container.label(relation), kg_container.definition(relation)])
