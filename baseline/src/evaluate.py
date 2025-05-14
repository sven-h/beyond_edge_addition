from argparse import ArgumentParser

from rdflib import Graph, RDFS, URIRef, BNode
from collections import defaultdict, Counter
import networkx as nx
import gzip
import json

from tqdm import tqdm


def smart_open(filename, mode='rt', encoding='utf-8'):
    if filename.endswith('.gz'):
        return gzip.open(filename, mode=mode, encoding=None if 'b' in mode else encoding)
    else:
        return open(filename, mode=mode, encoding=None if 'b' in mode else encoding)

def load_graph(filename):
    g = Graph()
    with smart_open(filename, mode='rb') as f:
        g.parse(file=f, format="nt")
    return g

def load_dataset(filename):
    # load the jsonl or json file depending on the file extension (if ending with .gz, use gzip)
    if filename.endswith('.json.gz') or filename.endswith('.json'):
        with smart_open(filename, mode='rt', encoding='utf-8') as f:
            for line in json.load(f):
                yield line
    elif filename.endswith('.jsonl.gz') or filename.endswith('.jsonl'):
        with smart_open(filename, mode='rt', encoding='utf-8') as f:
            for line in f:
                yield json.loads(line)
    else:
        raise ValueError(f"Unsupported file format: {filename}")


def make_rdflib_graph_from_list(triples, blank_prefix: str='_:'):
    g = Graph()
    blank_nodes = set()
    kg_nodes = set()
    new_triples = []
    for s, p, o in triples:
        new_s = BNode(s[len(blank_prefix):]) if s.startswith(blank_prefix) else URIRef(s)
        new_p = BNode(p[len(blank_prefix):]) if p.startswith(blank_prefix) else URIRef(p)
        new_o = BNode(o[len(blank_prefix):]) if o.startswith(blank_prefix) else URIRef(o)
        g.add((new_s, new_p, new_o))
        new_triples.append((str(new_s), str(new_p), str(new_o)))
        if isinstance(new_s, BNode):
            blank_nodes.add(str(new_s))
        else:
            kg_nodes.add(str(new_s))
        if isinstance(new_p, BNode):
            blank_nodes.add(str(new_p))
        else:
            kg_nodes.add(str(new_p))
        if isinstance(new_o, BNode):
            blank_nodes.add(str(new_o))
        else:
            kg_nodes.add(str(new_o))
    return g, new_triples, blank_nodes, kg_nodes

# properties_graph = load_graph("properties.nt.gz")
# class_graph = load_graph("classes.nt.gz")
#
# subclass_graph = nx.DiGraph()
# for i, (s, p, o) in enumerate(class_graph.triples((None, RDFS.subClassOf, None))):
#     subclass_graph.add_edge(s, o)


def get_blank_node_counter(ground_truth_graph, system_rdf_graph):
    blank_node_counter = defaultdict(Counter)
    for s, p, o in ground_truth_graph:
        if isinstance(s, BNode):
            # sanity check
            if isinstance(p, BNode) or isinstance(o, BNode):
                raise ValueError(f"Invalid triple with two blank nodes in ground truth: {s}, {p}, {o}")
            for system_s, system_p, system_o in system_rdf_graph.triples((None, p, o)):
                if isinstance(system_s, BNode):
                    blank_node_counter[str(s)][str(system_s)] += 1
        if isinstance(p, BNode):
            # sanity check
            if isinstance(s, BNode) or isinstance(o, BNode):
                raise ValueError(f"Invalid triple with two blank nodes in ground truth: {s}, {p}, {o}")
            for system_s, system_p, system_o in system_rdf_graph.triples((s, None, o)):
                if isinstance(system_p, BNode):
                    blank_node_counter[str(p)][str(system_p)] += 1

        if isinstance(o, BNode):
            # sanity check
            if isinstance(s, BNode) or isinstance(p, BNode):
                raise ValueError(f"Invalid triple with two blank nodes in ground truth: {s}, {p}, {o}")
            for system_s, system_p, system_o in system_rdf_graph.triples((s, p, None)):
                if isinstance(system_o, BNode):
                    blank_node_counter[str(o)][str(system_o)] += 1
    return blank_node_counter


def get_overlap_for_graph(set_of_entities_ground_truth, set_of_entities_system, graph):
    # get all reachable nodes from the set of entities in ground truth
    reachable_nodes_ground_truth = set()
    for entity in set_of_entities_ground_truth:
        reachable_nodes_ground_truth.add(entity)
        reachable_nodes_ground_truth.update(nx.descendants(graph, entity))

    reachable_nodes_system = set()
    for entity in set_of_entities_system:
        reachable_nodes_system.add(entity)
        reachable_nodes_system.update(nx.descendants(graph, entity))

    # get the intersection of the reachable nodes
    intersection = reachable_nodes_ground_truth.intersection(reachable_nodes_system)
    intersection_percentage = len(intersection) / len(
        reachable_nodes_ground_truth) if reachable_nodes_ground_truth else 0.0

    return intersection_percentage

def compute_metrics(tp, fp, fn):
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0
    f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0
    return recall, precision, f1

def update_counters(counters, intersecting, diff, diff_inverse, kg_nodes, prefix):
    counters[prefix + '_tp'] += len(intersecting)
    counters[prefix + '_fp'] += len(diff)
    counters[prefix + '_fn'] += len(diff_inverse)

    for x in intersecting:
        sub_prefix = 'kg_' if x in kg_nodes else 'nil_'
        counters[sub_prefix + prefix + '_tp'] += 1
    for x in diff:
        sub_prefix = 'kg_' if x in kg_nodes else 'nil_'
        counters[sub_prefix + prefix + '_fp'] += 1
    for x in diff_inverse:
        sub_prefix = 'kg_' if x in kg_nodes else 'nil_'
        counters[sub_prefix + prefix + '_fn'] += 1

def get_metrics(counters, label):
    recall, precision, f1 = compute_metrics(counters[label + '_tp'], counters[label + '_fp'],
                                            counters[label + '_fn'])
    return {
        f'{label}_recall': recall,
        f'{label}_precision': precision,
        f'{label}_f1': f1,
    }
def calculate_global_metrics(ground_truth, system, system_prefix):
    blank_node_counter = defaultdict(Counter)
    parsed_ground_truth = []
    parsed_system = []
    blank_nodes = set()
    kg_nodes = set()
    relation_counter = defaultdict(int)
    for idx, (ground_truth_item, system_item) in tqdm(enumerate(zip(ground_truth, system))):
        ground_truth_triples = ground_truth_item['triplets']
        system_triples = system_item['triplets'] if 'triplets' in system_item else system_item['linked_triplets']

        ground_truth_graph, triples, blank_nodes_, kg_nodes_ = make_rdflib_graph_from_list(ground_truth_triples)
        parsed_ground_truth.append(triples)
        blank_nodes.update(blank_nodes_)
        kg_nodes.update(kg_nodes_)
        # ground_truth_additional_info_graph = make_rdflib_graph_from_list(ground_truth_additional_info_triples)
        system_rdf_graph, triples, blank_nodes_, kg_nodes_ = make_rdflib_graph_from_list(system_triples, system_prefix)
        for node in blank_nodes_:
            if "relation" in node:
                relation_counter[node] += 1
        blank_nodes.update(blank_nodes_)
        kg_nodes.update(kg_nodes_)
        parsed_system.append(triples)

        # first step: find a best mapping between all blank nodes in the ground truth and the system
        blank_node_counter_ = get_blank_node_counter(ground_truth_graph, system_rdf_graph)
        for k, v in blank_node_counter_.items():
            for k2, v2 in v.items():
                blank_node_counter[k][k2] += v2

    blank_node_mapping = {}
    pairs = []
    for k, v in blank_node_counter.items():
        for k2, v2 in v.items():
            pairs.append(((k, k2), v2))
    pairs.sort(key=lambda x: x[1], reverse=True)

    gt_blank_nodes_selected = set()
    for (gt_blank_node, system_blank_node), _ in pairs:
        if gt_blank_node in gt_blank_nodes_selected:
            continue
        if system_blank_node in blank_node_mapping:
            continue
        blank_node_mapping[system_blank_node] = gt_blank_node

    # Initialize counters
    counters = defaultdict(int)

    for ground_truth_triples, system_triples in zip(parsed_ground_truth, parsed_system):
        mapped_system_triples = [
            (blank_node_mapping.get(s, s), blank_node_mapping.get(p, p), blank_node_mapping.get(o, o))
            for s, p, o in system_triples
        ]

        gt_triples = {tuple(x) for x in ground_truth_triples}
        gt_entities = {s for s, _, o in gt_triples} | {o for _, _, o in gt_triples}
        gt_relations = {p for _, p, _ in gt_triples}

        system_triples_set = {tuple(x) for x in mapped_system_triples}
        system_entities = {s for s, _, o in system_triples_set} | {o for _, _, o in system_triples_set}
        system_relations = {p for _, p, _ in system_triples_set}

        # Triples
        counters['triple_tp'] += len(gt_triples & system_triples_set)
        counters['triple_fp'] += len(system_triples_set - gt_triples)
        counters['triple_fn'] += len(gt_triples - system_triples_set)

        # Entities
        update_counters(counters, gt_entities & system_entities, system_entities - gt_entities,
                        gt_entities - system_entities, kg_nodes, 'entity')

        # Relations
        update_counters(counters, gt_relations & system_relations, system_relations - gt_relations,
                        gt_relations - system_relations, kg_nodes, 'relation')

    # Print all metrics
    all_metrics = {}
    for label in ['triple', 'entity', 'kg_entity', 'nil_entity', 'relation', 'kg_relation', 'nil_relation']:
        all_metrics.update(get_metrics(counters, label))
    return all_metrics


def calculate_local_metrics(ground_truth, system, system_prefix):
    parsed_ground_truth = []
    parsed_system = []
    blank_nodes = set()
    kg_nodes = set()
    more_triples_created = 0
    fewer_triples_created = 0
    exact_triples_created = 0
    for idx, (ground_truth_item, system_item) in tqdm(enumerate(zip(ground_truth, system))):
        ground_truth_triples = ground_truth_item['triplets']
        system_triples = system_item['triplets'] if 'triplets' in system_item else system_item['linked_triplets']

        ground_truth_graph, ground_truth_triples, blank_nodes_, kg_nodes_ = make_rdflib_graph_from_list(ground_truth_triples)
        blank_nodes.update(blank_nodes_)
        kg_nodes.update(kg_nodes_)
        # ground_truth_additional_info_graph = make_rdflib_graph_from_list(ground_truth_additional_info_triples)
        system_rdf_graph, system_triples, blank_nodes_, kg_nodes_ = make_rdflib_graph_from_list(system_triples, system_prefix)
        if len(system_triples) > len(ground_truth_triples):
            more_triples_created += 1
        elif len(system_triples) < len(ground_truth_triples):
            fewer_triples_created += 1
        else:
            exact_triples_created += 1
        blank_nodes.update(blank_nodes_)
        kg_nodes.update(kg_nodes_)


        # first step: find a best mapping between all blank nodes in the ground truth and the system
        blank_node_counter = get_blank_node_counter(ground_truth_graph, system_rdf_graph)
        blank_node_mapping = {}
        pairs = []
        for k, v in blank_node_counter.items():
            for k2, v2 in v.items():
                pairs.append(((k, k2), v2))
        pairs.sort(key=lambda x: x[1], reverse=True)

        gt_blank_nodes_selected = set()
        for (gt_blank_node, system_blank_node), _ in pairs:
            if gt_blank_node in gt_blank_nodes_selected:
                continue
            if system_blank_node in blank_node_mapping:
                continue
            blank_node_mapping[system_blank_node] = gt_blank_node

        mapped_system_triples = [
            (blank_node_mapping.get(s, s), blank_node_mapping.get(p, p), blank_node_mapping.get(o, o))
            for s, p, o in system_triples
        ]

        parsed_system.append(mapped_system_triples)
        parsed_ground_truth.append(ground_truth_triples)




    # Initialize counters
    counters = defaultdict(int)
    for ground_truth_triples, system_triples in zip(parsed_ground_truth, parsed_system):
        gt_triples = {tuple(x) for x in ground_truth_triples}
        system_triples_set = {tuple(x) for x in system_triples}

        gt_entities = {s for s, _, o in gt_triples} | {o for _, _, o in gt_triples}
        gt_relations = {p for _, p, _ in gt_triples}

        system_entities = {s for s, _, o in system_triples_set} | {o for _, _, o in system_triples_set}
        system_relations = {p for _, p, _ in system_triples_set}

        # Triples
        counters['triple_tp'] += len(gt_triples & system_triples_set)
        counters['triple_fp'] += len(system_triples_set - gt_triples)
        counters['triple_fn'] += len(gt_triples - system_triples_set)

        # Entities
        update_counters(counters, gt_entities & system_entities, system_entities - gt_entities,
                        gt_entities - system_entities, kg_nodes, 'entity')

        # Relations
        update_counters(counters, gt_relations & system_relations, system_relations - gt_relations,
                        gt_relations - system_relations, kg_nodes, 'relation')

        # Print all metrics
    all_metrics = {"more_triples_created": more_triples_created/len(ground_truth),
                   "fewer_triples_created": fewer_triples_created/len(ground_truth),
                   "exact_triples_created": exact_triples_created/len(ground_truth),}
    for label in ['triple', 'entity', 'kg_entity', 'nil_entity', 'relation', 'kg_relation', 'nil_relation']:
        all_metrics.update(get_metrics(counters, label))
    return all_metrics



def evaluate(system, ground_truth, system_prefix: str = "http://createdbyedc.org/"):
    assert len(ground_truth) == len(system)
    local_metrics = calculate_local_metrics(ground_truth, system, system_prefix)
    print(json.dumps(local_metrics, indent=4))
    print("-------------------------------")
    global_metrics = calculate_global_metrics(ground_truth, system, system_prefix)
    print(json.dumps(global_metrics, indent=4))

    return global_metrics, local_metrics



if __name__ == "__main__":
    argparser = ArgumentParser()
    argparser.add_argument("--ground_truth", type=str, default="test_dataset.jsonl")
    argparser.add_argument("--system", type=str, default="final_linked_results.json")
    argparser.add_argument("--system_prefix", type=str, default="http://createdbyedc.org/")

    args = argparser.parse_args()
    ground_truth = load_dataset(args.ground_truth)
    system = load_dataset(args.system)
    ground_truth = list(ground_truth)
    system = list(system)
    evaluate(system, ground_truth, system_prefix=args.system_prefix)

