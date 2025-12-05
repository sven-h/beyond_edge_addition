from rdflib_hdt import HDTStore
from rdflib import Graph, URIRef
from rdflib import RDFS, SKOS, Literal
import pickle
import csv

store = HDTStore("/dev/shm/wikidata-20251027-all-BETA.hdt")
graph = Graph(store=store)

with open("kg_info.pickle", 'rb') as f:
    object_to_load = pickle.load(f)
    classes_that_are_instances = object_to_load["classes_that_are_instances"]
    all_instances = object_to_load["all_instances"]
    real_classes = object_to_load["real_classes"]

def is_lang(my_literal):
    # check for type of my_literal to see if it is really a literal
    if not isinstance(my_literal, Literal):
        return True
    if my_literal.language == None:
        return True
    lower_lang = my_literal.language.lower()
    if lower_lang == "en" or lower_lang == "mul":
        return True
    if lower_lang.startswith("en-"):
        return True
    return False


def write(properties, corpus_name):
    seen_instances = set()
    seen_classes = set()
    with open(f"{corpus_name}_instances.csv", 'w', newline="", encoding="utf-8") as instance_file,\
         open(f"{corpus_name}_classes.csv", 'w', newline="", encoding="utf-8") as classes_file:
        writer_instances = csv.writer(instance_file)
        writer_classes = csv.writer(classes_file)
        for selected_property in properties:
            for s, p, o in graph.triples((None, selected_property, None)):
                if is_lang(o):
                    if (s in all_instances or s in classes_that_are_instances):
                        key = (str(s),str(o))
                        if key not in seen_instances:
                            seen_instances.add(key)
                            writer_instances.writerow([str(s),str(o)])
                    if s in real_classes:
                        key = (str(s),str(o))
                        if key not in seen_classes:
                            seen_classes.add(key)
                            writer_classes.writerow([str(s),str(o)])

print("start")
write([RDFS.label, SKOS.altLabel], "label")
write([URIRef("http://schema.org/description")], "description")
print("end")