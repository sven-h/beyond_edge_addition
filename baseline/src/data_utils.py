import dataclasses
import json
import os.path
import pickle
from collections import defaultdict
from typing import List, TypedDict, Dict, Optional
from re import match

import jsonlines
from tqdm import tqdm


@dataclasses.dataclass
class EntityDict:
    qid: str
    candidate_mention: Optional[str] = None
    candidate_definition: Optional[str] = None
    candidate_types: Optional[List[str]] = None

@dataclasses.dataclass
class KGElement:
    identifier: str = ""
    label: str = ""
    definition: str = ""
    aliases: List[str] = dataclasses.field(default_factory=list)
    types: List[str] = dataclasses.field(default_factory=list)

@dataclasses.dataclass
class Example:
    text: str
    entities: List[EntityDict]
    triples: List[tuple]


def create_full_entity_description(label: str, definition: str, types: List[str]) -> str:
    """
    Create a full entity definition by combining the label, definition, and types.
    """
    types = [type_ for type_ in types if isinstance(type_, str) and type_ != ""]
    if len(types) > 0:
        return f"{label} ({', '.join(types[:3])}): {definition}"
    else:
        return f"{label}: {definition}"


def get_raw_text(text: str) -> tuple:
    lang_tag = ""
    if "@" in text:
        position = text.rfind("@")
        lang_tag = text[position + 1:]
        text = text[:position]
    return text, lang_tag

class KGContainer:
    def __init__(self, data_folder: str = "data/ie_dataset_v3/"):
        if os.path.exists(data_folder + "cached.pkl"):
            relations, classes, entities, entity_qids = pickle.load(open(data_folder + "cached.pkl", "rb"))
            self.relations = relations
            self.classes = classes
            self.entities = entities
            self.entity_qids = entity_qids
        else:
            classes_file = data_folder + "classes.nt"
            entities_file = data_folder + "entities.nt"
            properties_file = data_folder + "properties.nt"

            relations = self.init_dicts(properties_file)
            self.relations = relations
            classes = self.init_dicts(classes_file)
            classes['http://www.w3.org/2002/07/owl#Class'] = KGElement(
                identifier='http://www.w3.org/2002/07/owl#Class',
                label='concept',
                definition='semantic unit understood in different ways, e.g. as mental representation, ability or abstract object (philosophy)',
                aliases=[],
                types=[]
            )
            self.classes = classes
            self.entities = self.init_dicts(entities_file)
            self.entity_qids = set(self.entities.keys())
            self.entities.update(relations)
            self.entities.update(classes)

            # Save the dictionaries to a pickle file for future use
            with open(data_folder + "cached.pkl", "wb") as f:
                pickle.dump((self.relations, self.classes, self.entities, self.entity_qids), f)



    @staticmethod
    def normalize(element: str) -> str:
        element = element.strip()
        uri_regex = r"<(.*?)>"
        if match(uri_regex, element):
            return element[1:-1]
        # Remove language tags and quotes
        element = element.replace("@en", "").replace('"', "")
        return element




    def init_dicts(self, filename: str) -> dict:
        print(f"Loading {filename}...")
        labels_dict = defaultdict(set)
        descriptions_dict = defaultdict(set)
        dictionary = defaultdict(KGElement)
        for line in tqdm(open(filename, "r", encoding="utf-8")):
            line = line[:-2]
            split_line = line.split(" ")
            sub = self.normalize(split_line[0])
            pred = self.normalize(split_line[1])
            obj = self.normalize(" ".join(split_line[2:]))
            dictionary[sub].identifier= sub
            if pred == 'http://www.w3.org/1999/02/22-rdf-syntax-ns#type':
                dictionary[sub].types.append(obj)
            elif pred == 'http://www.w3.org/2000/01/rdf-schema#label':
                labels_dict[sub].add(obj)
            elif pred == 'http://www.w3.org/2000/01/rdf-schema#comment':
                descriptions_dict[sub].add(obj)
            elif pred == "http://www.w3.org/2004/02/skos/core#altLabel":
                dictionary[sub].aliases.append(obj)
        for key, value in dictionary.items():
            definitions = descriptions_dict[key]
            labels = labels_dict[key]
            for definition in definitions:
                definition, lang_tag = get_raw_text(definition)
                if lang_tag == "en" or lang_tag == "":
                    dictionary[key].definition = definition
                    break
            else:
                definition, _ = get_raw_text(next(iter(definitions), ""))
                dictionary[key].definition = definition
            for label in labels:
                label, lang_tag = get_raw_text(label)
                if lang_tag == "en" or lang_tag == "":
                    dictionary[key].label = label
                    break
            else:
                label, _ = get_raw_text(next(iter(labels), ""))
                dictionary[key].label = label
        dictionary = {k: v for k, v in dictionary.items() if len(v.label) > 0}
        return dictionary


    def types(self, qid: str, return_raw: bool = False) -> List[str]:
        if return_raw:
            return self.entities[qid].types
        else:
            type_strings = [self.label(type_) for type_ in self.entities[qid].types if type_ in self.entities]
            type_strings = [x for x in  type_strings if x != ""]
        return type_strings

    def definition(self, qid: str) -> str:
        return self.entities[qid].definition

    def label(self, qid: str) -> str:
        return self.entities[qid].label

    def aliases(self, qid: str) -> List[str]:
        return self.entities[qid].aliases


def load_data(file_name: str, kgc: KGContainer) -> List[Example]:
    examples = []
    missing_entities = 0
    num_entities = 0
    for item in tqdm(open(file_name)):
        raw_example = json.loads(item)
        if "triplets" in raw_example:
            entities = set()
            for s, p, o in raw_example["triplets"]:
                entities.add(s)
                entities.add(o)
            entities = [EntityDict(qid=entity) for entity in entities]
            example = Example(text=raw_example["text"], triples=raw_example["triplets"], entities=entities)
        else:
            entities = []
            generated_entities = {x["entity"]: x for x in raw_example["generated_entities"]}
            for entity in raw_example["entities"]:
                if entity["qid"] not in kgc.entities:
                    print(f"Entity {entity['qid']} not found in KGContainer.")
                    continue
                label = kgc.label(entity["qid"])
                if label in generated_entities:
                    try:
                        surface_form = generated_entities[label]["surface_form"]
                        if isinstance(surface_form, list):
                            if len(surface_form) > 0:
                                surface_form = surface_form[0]
                            else:
                                surface_form = None
                        definition = generated_entities[label]["definition"]
                        if isinstance(definition, list):
                            definition = definition[0]
                        type_s = generated_entities[label].get("types", generated_entities[label].get("type", None))
                        if isinstance(type_s, list):
                            if len(type_s) > 0:
                                type_s = type_s[0]
                            else:
                                type_s = None
                        entities.append(EntityDict(qid=entity["qid"],
                                                candidate_mention=surface_form,
                                                candidate_definition=definition,
                                                candidate_types=[type_s]))
                    except KeyError:
                        print(f"KeyError for entity {entity['qid']}")
                        print("Skipping entity due to missing data.")
                        continue
                else:
                    entities.append(EntityDict(qid=entity["qid"]))
                    missing_entities += 1

            example = Example(text=raw_example["text"], triples=raw_example["triples"], entities=entities)
        num_entities += len(entities)
        examples.append(example)
    print(f"Ratio of missing entities: {missing_entities / num_entities}")
    return examples




if __name__ == "__main__":
    kgc = KGContainer()
    examples = load_data("outputs.jsonl", kgc) # "data/filtered_dataset.json")
