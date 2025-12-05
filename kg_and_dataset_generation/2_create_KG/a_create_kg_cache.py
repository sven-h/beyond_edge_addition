# All imports

from rdflib_hdt import HDTStore
from rdflib import Graph, URIRef, RDF
from datetime import datetime
import networkx as nx
import pickle


store = HDTStore("/dev/shm/wikidata-20251027-all-BETA.hdt")


# Display some metadata about the HDT document itself
print(f"Number of RDF triples: {len(store)}")
print(f"Number of subjects: {store.nb_subjects}")
print(f"Number of predicates: {store.nb_predicates}")
print(f"Number of objects: {store.nb_objects}")

# Create an RDFlib Graph with the HDT document as a backend
graph = Graph(store=store)


all_classes = set()
all_instances = set()

#  class-item is any item used as value in a instance of (P31) statement
for i, (s, p, o) in enumerate(graph.triples((None, URIRef("http://www.wikidata.org/prop/direct/P31"), None))):
    all_instances.add(s)
    all_classes.add(o)
    if i % 1000000 == 0:
        print(f"{datetime.now().time()} - iterate over 'instance of' statements: {i}")
# ~ 10 min - 20 min



# class-item is any item used as value or object in a subclass of (P279) statement
subclass_graph = nx.DiGraph()
for i, (s, p, o) in enumerate(graph.triples((None, URIRef("http://www.wikidata.org/prop/direct/P279"), None))):
    all_classes.add(s)
    all_classes.add(o)
    subclass_graph.add_edge(s, o)
    if i % 1000000 == 0:
        print(f"{datetime.now().time()} - iterate over 'subclass of' statements: {i} / ~ 4000000")
print(f"all classes: {len(all_classes)}")
print(f"all instances: {len(all_instances)}")
# ~ 6 min



# filter out classes that can be instances (e.g. classes that do not have any instances and are no superclass of other classes)
classes_that_are_instances = set()

class_leaves = [node for node in subclass_graph.nodes if subclass_graph.in_degree(node) == 0]
print(f"class leaves (no ingoing subclass properties): {len(class_leaves)}")
for clazz in class_leaves:
    has_instances = False
    for s,p,o in graph.triples((None, URIRef("http://www.wikidata.org/prop/direct/P31"), clazz)):
        has_instances = True
        break
    if not has_instances:
        classes_that_are_instances.add(clazz)
print(f"classes that are instances: {len(classes_that_are_instances)}")
real_classes = all_classes - classes_that_are_instances
print(f"real classes: {len(real_classes)}")

for class_to_be_instance in classes_that_are_instances:
    subclass_graph.remove_node(class_to_be_instance)
    
    
    
# fix all cycles
while True:
    try:
        cycle = nx.find_cycle(subclass_graph, orientation="original")
        print(cycle)
        subclass_graph.remove_edge(cycle[0][0], cycle[0][1])
    except nx.exception.NetworkXNoCycle:
        break
# ~ 65 min




all_properties = set()
for s, p, o in graph.triples((None, URIRef("http://wikiba.se/ontology#propertyType"), None)):
    all_properties.add(s)
for s, p, o in graph.triples((None, RDF.type, URIRef("http://wikiba.se/ontology#Property"))):
    all_properties.add(s)

subproperty_graph = nx.DiGraph()
for i, (s, p, o) in enumerate(graph.triples((None, URIRef("http://www.wikidata.org/prop/direct/P1647"), None))): # subproperty of 
    subproperty_graph.add_edge(s, o)
for i, (s, p, o) in enumerate(graph.triples((None, URIRef("http://www.wikidata.org/prop/direct/P2236"), None))): # external subproperty of
    subproperty_graph.add_edge(s, o)


# search for properties that are identifiers:
# https://www.wikidata.org/wiki/Wikidata:Identifiers
# Wikidata property with datatype external identifier: ?p wikibase:propertyType wikibase:ExternalId
identifier_properties = set()
property_type = URIRef("http://wikiba.se/ontology#propertyType")
external_id = URIRef("http://wikiba.se/ontology#ExternalId")
for my_prop in all_properties:
    if (my_prop, property_type, external_id) in graph:
        identifier_properties.add(my_prop)
non_identifier_properties = all_properties - identifier_properties

non_identifier_properties_direct = set()
for p in non_identifier_properties:
    for s, p, o in graph.triples((p, URIRef("http://wikiba.se/ontology#directClaim"), None)):
        non_identifier_properties_direct.add(o)

print(f"Number of non-identifer properties: {len(non_identifier_properties_direct)} (general uri props: {len(non_identifier_properties)}) out of {len(all_properties)}")




# fix all cycles
while True:
    try:
        cycle = nx.find_cycle(subproperty_graph, orientation="original")
        print(cycle)
        subproperty_graph.remove_edge(cycle[0][0], cycle[0][1])
    except nx.exception.NetworkXNoCycle:
        break
# ~


# remove classes and properties from instances
all_instances = all_instances - real_classes
all_instances = all_instances - classes_that_are_instances # those are handeled differently troughout the programm
all_instances = all_instances - all_properties


kg_cache_file = "kg_info.pickle"
with open(kg_cache_file, 'wb') as f:
    object_to_store = {
        "non_identifier_properties_direct": non_identifier_properties_direct,
        "classes_that_are_instances": classes_that_are_instances,
        "real_classes": real_classes,
        "all_instances": all_instances,
        "subclass_graph": subclass_graph,
        "subproperty_graph": subproperty_graph
    }
    pickle.dump(object_to_store, f)