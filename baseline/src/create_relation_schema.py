import csv

from src.data_utils import KGContainer

if __name__ == "__main__":
    kg_container = KGContainer()
    writer = csv.writer(open("relation_schema.csv", "w"))
    for relation in kg_container.relations:
        writer.writerow([relation, kg_container.label(relation), kg_container.definition(relation)])
