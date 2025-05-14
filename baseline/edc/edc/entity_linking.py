import json
import logging
import random
from collections import defaultdict
from typing import List, Optional, Iterable

import math
import networkx as nx
from sentence_transformers import SentenceTransformer, CrossEncoder
from sklearn.preprocessing import normalize
from tqdm import tqdm
from vllm import LLM

import edc.utils.llm_utils as llm_utils
from transformers import AutoModelForCausalLM, AutoTokenizer, AutoModel
import numpy as np
import faiss

from src.create_entity_linking_dataset import formulate_candidates
from src.data_utils import create_full_entity_description
from src.special_prompts import STS_PROMPT

logger = logging.getLogger(__name__)

class EntityLinker:
    # The class to handle the last stage: Schema Canonicalization
    def __init__(self, sentence_encoder: SentenceTransformer,
                 entity_index: faiss.Index, index_to_entity: dict , internal_entities: list,
                 model: AutoModelForCausalLM = None, tokenizer: AutoTokenizer = None, openai_model=None, disambiguator_model: CrossEncoder=None, cluster=False,
                 me_threshold=0.5, mm_threshold=0.5, path_threshold=0.75,
                 ) -> None:
        # The canonicalizer uses an embedding model to first fetch candidates from the target schema, then uses a verifier schema to decide which one to canonicalize to or not
        # canonoicalize at all.

        assert openai_model is not None or (model is not None and tokenizer is not None)
        if isinstance(model, tuple):
            self.model, self.adapter = model
        else:
            self.model = model
            self.adapter = None
        if disambiguator_model is not None:
            self.disambiguator_model = disambiguator_model
        else:
            self.disambiguator_model = None
        self.tokenizer = tokenizer
        self.me_threshold = me_threshold
        self.cluster = cluster
        self.mm_threshold = mm_threshold
        self.path_threshold = path_threshold
        self.openai_model = openai_model
        self.sentence_encoder = sentence_encoder
        self.entity_index = entity_index
        self.index_to_entity = index_to_entity
        self.entity_disambiguation_prompt = open("./data/entity_disambiguation_prompt.txt").read()
        self.finetuned_entity_disambiguation_prompt = open("./data/finetuned_entity_disambiguation_prompt.txt").read()
        self.entity_description_prompt = open("./data/entity_description_prompt.txt").read()
        self.entity_description_few_shot_examples = open("./data/entity_description_few_shot_examples.txt").read()
        # self.entity_disambiguation_few_shot_examples = open("data/entity_disambiguation_few_shot_examples.txt").read()
        self.internal_entity_reps = self.initialize_internal_reps(internal_entities)

    def initialize_internal_reps(self, internal_entities: List[dict]) -> dict:
        if len(internal_entities) == 0:
            return {}
        all_encodings = self.sentence_encoder.encode([elem["rep"] for elem in internal_entities], prompt_name="sts_query", normalize_embeddings=True)
        internal_entity_reps = {}
        for elem, query_embedding in zip(internal_entities, all_encodings):
            internal_entity_reps[elem["entity"]] = {"encoding": query_embedding,
                                                    **elem}
        return internal_entity_reps


    def generate_candidates(self, definition_encodings: list, top_k=10):
        target_entities = list(self.internal_entity_reps.keys())
        # target_entity_encodings = [self.internal_entity_reps[entity]["encoding"] for entity in target_entities]
        target_entity_encodings = []
        encoding_indices = []
        for idx, entity in enumerate(target_entities):
            for idx_2, encoding in enumerate(self.internal_entity_reps[entity]["encodings"]):
                target_entity_encodings.append(encoding)
                encoding_indices.append((idx, idx_2))
        query_encodings = np.array(definition_encodings)
        candidates_per_entity = []
        if len(target_entity_encodings) > 0:
            all_internal_scores = np.array(query_encodings) @ np.array(target_entity_encodings).T
        else:
            all_internal_scores = np.zeros((query_encodings.shape[0], len(target_entity_encodings)))
        if self.entity_index is None:
            all_kg_indices = [[]] * query_encodings.shape[0]
        else:
            all_kg_indices = self.entity_index.search(query_encodings, top_k)[1]
        for internal_scores, kg_indices in zip(all_internal_scores, all_kg_indices):
            top_k_internal_indices = np.argsort(-internal_scores)

            kg_entities = [self.index_to_entity[i] for i in kg_indices]
            internal_entities = [(self.internal_entity_reps[target_entities[encoding_indices[i][0]]], encoding_indices[i][1]) for i in top_k_internal_indices[:top_k]]
            candidates_per_entity.append(
                {
                    "internal_entities": internal_entities,
                    "kg_entities": kg_entities,
                }
            )

        return candidates_per_entity

    def parse_candidate_disambiguation(self, completion: str, candidates: list):
        for idx in reversed(range(len(candidates))):
            if f"{idx + 1}" in completion:
                return candidates[idx]
        return None

    def parse_ft_candidate_disambiguation(self, completion: str, candidates: list):
        start_bracket = completion.find("{")
        end_bracket = completion.rfind("}")
        snippet = completion[start_bracket:end_bracket + 1]
        try:
            completion = json.loads(snippet)
            return candidates[int(completion["ID"])]
        except:
            return None

    def generate(self, all_messages: List[list], deactivate_pbar=False):
        if self.openai_model is None:
            completions = llm_utils.generate_completion_transformers(
                        all_messages, self.model, self.tokenizer, answer_prepend="", deactivate_pbar=deactivate_pbar
                    )
        else:
            completions = []
            for messages  in all_messages:
                completion = llm_utils.openai_chat_completion(self.openai_model, None, messages)
                completions.append(completion)
        return completions

    def ce_rep(self, name, description, types):
        candidate_types = [x for x in types if isinstance(x, str)]
        candidate_types_str = ", ".join(candidate_types)
        return f"{name} ({candidate_types_str}): {description}"
    def ce_disambiguate(self, definitions, candidates):
        sentence_pairs = []
        idx_to_pairs = []
        for definition, candidate_container in zip(definitions, candidates):
            main_rep = self.ce_rep(definition["entity"],
                                                    definition["definition"],
                                                    [definition["type"]])
            pair_indices = []
            for internal_entity, specific_example_idx in candidate_container["internal_entities"]:
                candidate = internal_entity["samples"][specific_example_idx]
                pair_indices.append(len(sentence_pairs))
                sentence_pairs.append(
                    (main_rep,
                     self.ce_rep(candidate["entity"],
                                                    candidate["definition"],
                                                    [candidate["type"]]))
                )
            for candidate in candidate_container["kg_entities"]:
                pair_indices.append(len(sentence_pairs))
                sentence_pairs.append(
                    (main_rep,
                     self.ce_rep(candidate["label"],
                                                    candidate["definition"],
                                                    candidate["types"]))
                )
            idx_to_pairs.append(pair_indices)
        if sentence_pairs:
            scores = self.disambiguator_model.predict(sentence_pairs)
        else:
            return [None for _ in range(len(definitions))]

        linked_entities = []
        for definition, candidate_container, eligible_pairs in zip(definitions, candidates, idx_to_pairs):
            max_score = -1
            best_candidate = None
            all_candidates = candidate_container["internal_entities"] + candidate_container["kg_entities"]
            for idx, pair_idx in enumerate(eligible_pairs):
                threshold = self.me_threshold if idx < len(candidate_container["internal_entities"]) else self.mm_threshold
                if scores[pair_idx] > max_score and scores[pair_idx] > threshold:
                    max_score = scores[pair_idx]
                    best_candidate = all_candidates[idx]
            if isinstance(best_candidate, tuple):
                best_candidate = best_candidate[0]
            linked_entities.append(best_candidate)
        return linked_entities


    def llm_disambiguate(self, definitions, candidates):
        all_messages = []
        for definition, candidate_container in zip(definitions, candidates):
            if self.adapter:
                tmp_rep = [{"title": entity["entity"],
                            "entity_description": entity["definition"],
                            "entity_types": [entity["type"]]
                            } for entity in candidate_container["internal_entities"]]
                tmp_rep += [{"title": entity["label"],
                             "entity_description": entity["definition"],
                             "entity_types": entity["types"]
                             } for entity in candidate_container["kg_entities"]]
                candidate_str = formulate_candidates(tmp_rep, 256)
                filled_prompt = self.entity_disambiguation_prompt.format_map(
                    {
                        "entity": definition["entity"],
                        "definition": definition["definition"],
                        "types": [definition["type"]],
                        "entity_candidates": candidate_str}
                )
            else:
                entity_candidates_str = ""
                counter = 0
                for candidate in candidate_container["internal_entities"]:
                    entity_candidates_str += f"{counter + 1}: " + create_full_entity_description(
                        candidate["entity"],
                        candidate["definition"],
                        [candidate["type"]]) + "\n"
                    counter += 1

                for candidate in candidate_container["kg_entities"]:
                    entity_candidates_str += f"{counter + 1}: " + create_full_entity_description(
                        candidate["label"],
                        candidate["definition"],
                        candidate["types"]) + "\n"
                    counter += 1
                entity_candidates_str += f"{counter + 1}: None\n"

                filled_prompt = self.entity_disambiguation_prompt.format_map(
                    {"few_shot_examples": self.entity_description_few_shot_examples,
                     "entity_rep": create_full_entity_description(
                         definition["entity"],
                         definition["definition"],
                         [definition["type"]]),
                     "entity_candidates": entity_candidates_str}
                )
            messages = [{"role": "user", "content": filled_prompt}]
            all_messages.append(messages)
        completions = self.generate(all_messages, deactivate_pbar=True)
        linked_entities = []
        for candidate_container, completion in zip(candidates, completions):
            if self.adapter is not None:
                linked_entities.append(self.parse_ft_candidate_disambiguation(completion,
                                                                              candidate_container["internal_entities"] +
                                                                              candidate_container["kg_entities"]))

            else:
                linked_entities.append(self.parse_candidate_disambiguation(completion,
                                                                           candidate_container["internal_entities"] +
                                                                           candidate_container["kg_entities"]))
        return linked_entities

    def calculate_neighbors(self, all_definitions: List[List[dict]], all_query_encodings: List[list], all_definition_encodings: List[list], top_k=10):
        encoding_to_index = []
        full_query_array = []
        full_definition_array = []
        definitions_list = []
        full_indices = []
        for idx, (query_encodings, definition_encodings, definitions) in enumerate(zip(all_query_encodings, all_definition_encodings, all_definitions)):
            encoding_indices = []
            for idx_2, (encoding, definition_encoding, definition) in enumerate(zip(query_encodings, definition_encodings, definitions)):
                encoding_indices.append(len(full_query_array))
                full_query_array.append(encoding)
                full_definition_array.append(definition_encoding)
                definitions_list.append(definition)
                full_indices.append((idx, idx_2))
            encoding_to_index.append(encoding_indices)
        full_query_array = np.array(full_query_array)
        full_definition_array = np.array(full_definition_array)

        top_k_indices = self.batched_similarity_topk(full_query_array, full_definition_array)


        if self.entity_index is None:
            all_kg_indices = [[]] * full_query_array.shape[0]
        else:
            all_kg_indices = self.entity_index.search(full_query_array, top_k)[1]

        me_pairs_idents = []
        me_pairs = []
        mm_pairs_idents= []
        mm_pairs = []
        idx_to_pairs = []
        for definition, internal_indices, external_indices, (example_idx, entity_idx) in zip(definitions_list, top_k_indices, all_kg_indices, full_indices):
            main_rep = self.ce_rep(definition["entity"],
                                   definition["definition"],
                                   [definition["type"]])

            kg_entities =  [self.index_to_entity[i] for i in external_indices]
            pair_indices = []
            for internal_idx in internal_indices:
                other_example_idx, other_entity_idx = full_indices[internal_idx]
                candidate = all_definitions[other_example_idx][other_entity_idx]
                mm_pairs.append(
                    (main_rep,
                     self.ce_rep(candidate["entity"],
                                 candidate["definition"],
                                 [candidate["type"]]))
                )
                mm_pairs_idents.append(((example_idx, entity_idx), (other_example_idx, other_entity_idx)))
            for candidate in kg_entities:
                me_pairs.append(
                    (main_rep,
                     self.ce_rep(candidate["label"],
                                 candidate["definition"],
                                 candidate["types"]))
                )
                me_pairs_idents.append(((example_idx, entity_idx), candidate["identifier"]))
            idx_to_pairs.append(pair_indices)
        if me_pairs:
            me_scores = self.disambiguator_model.predict(me_pairs)
        else:
            me_scores = []
        if mm_pairs:
            mm_scores = self.disambiguator_model.predict(mm_pairs)
        else:
            mm_scores = []

        return zip(me_pairs_idents, me_scores), zip(mm_pairs_idents, mm_scores)



    def _get_alignment_graph(self, mention_entity_scores, mention_mention_scors) -> nx.Graph:
        ag = nx.Graph()
        for (m_id, e_id), score in mention_entity_scores:
            ag.add_node(m_id, is_ent=False)
            if True and score > self.me_threshold:
                ag.add_node(e_id, is_ent=True)
                ag.add_edge(m_id, e_id, weight=min(score, 1))
        ag.add_weighted_edges_from(
            [(u, v, min(score, 1)) for (u, v), score in mention_mention_scors if
             score > self.mm_threshold])
        return ag

    def _get_subgraphs(self, ag: nx.Graph) -> Iterable[nx.Graph]:
        for nodes in nx.connected_components(ag):
            yield ag.subgraph(nodes)

    @classmethod
    def _get_mention_nodes(cls, g: nx.Graph) -> set:
        return {node for node, is_ent in g.nodes(data='is_ent') if not is_ent}

    @classmethod
    def _get_entity_node(cls, g: nx.Graph) -> Optional[int]:
        ent_nodes = cls._get_entity_nodes(g)
        return ent_nodes[0] if ent_nodes else None

    @classmethod
    def _get_entity_nodes(cls, g: nx.Graph) -> List[int]:
        return [node for node, is_ent in g.nodes(data='is_ent') if is_ent]

    def _split_into_valid_subgraphs(self, ag: nx.Graph) -> List[nx.Graph]:
        ent_groups = defaultdict(set)
        unassigned_mentions = set()
        distances, paths = nx.multi_source_dijkstra(ag, self._get_entity_nodes(ag), weight=self._to_dijkstra_node_weight)
        for node, path in paths.items():
            score = self._from_dijkstra_node_weight(distances[node])
            if score > self.path_threshold:
                ent_node = path[0]
                ent_groups[ent_node].add(node)
            else:
                unassigned_mentions.add(node)
        return [ag.subgraph(nodes) for nodes in ent_groups.values()] + list(
            self._get_subgraphs(ag.subgraph(unassigned_mentions)))

    @staticmethod
    def _to_dijkstra_node_weight(u, v, attrs: dict) -> float:
        return -math.log2(attrs['weight'])
    @staticmethod
    def _from_dijkstra_node_weight(weight: float) -> float:
        return 2 ** (-weight)

    def _compute_valid_subgraphs(self, ag: nx.Graph) -> List[nx.Graph]:
        valid_subgraphs = []
        for sg in self._get_subgraphs(ag):
            if self._is_valid_graph(sg):
                valid_subgraphs.append(sg)
            else:
                valid_subgraphs.extend(self._split_into_valid_subgraphs(sg))
        return valid_subgraphs

    def _is_valid_graph(self, ag: nx.Graph) -> bool:
        return len(self._get_entity_nodes(ag)) <= 1

    def batched_similarity_topk(self, full_query_array, full_definition_array, top_k=10, batch_size=1024):
        num_queries = full_query_array.shape[0]
        topk_indices = []

        for start in range(0, num_queries, batch_size):
            end = min(start + batch_size, num_queries)
            query_batch = full_query_array[start:end]

            # Compute similarities for this batch
            similarities = np.dot(query_batch, full_definition_array.T)

            # Mask diagonal (self-similarity)
            row_indices = np.arange(start, end)
            similarities[np.arange(end - start), row_indices] = -1

            # Get top-k indices for this batch
            batch_topk = np.argsort(-similarities, axis=1)[:, :top_k]
            topk_indices.append(batch_topk)

        return np.vstack(topk_indices)

    def disambiguate_by_clustering(self, all_triples, all_definitions, mention_entity_scores, mention_mention_scores):
        ag = self._get_alignment_graph(mention_entity_scores, mention_mention_scores)
        valid_subgraphs = self._compute_valid_subgraphs(ag)
        clusters = [(self._get_mention_nodes(g), self._get_entity_node(g)) for g in valid_subgraphs]
        mapping = {}
        counter = 0
        for elements, identifier in clusters:
            if identifier is None:
                identifier =f"http://createdbyedc.org/entities/{counter}"
                counter += 1
            for doc_idx, entity_idx in elements:
                mapping[(doc_idx, entity_idx)] = identifier

        entity_mapped_triples = []
        for doc_idx, (triples, definitions) in enumerate(zip(all_triples, all_definitions)):
            entity_mapping = {}
            for entity_idx, definition in enumerate(definitions):
                entity_mapping[definition["entity"]] = mapping[(doc_idx, entity_idx)]
            mapped_triples = []
            for s, p, o in triples:
                subject_entity = entity_mapping[str(s)]
                object_entity = entity_mapping[str(o)]
                mapped_triples.append((subject_entity, p, object_entity))
            entity_mapped_triples.append(mapped_triples)


        return entity_mapped_triples
    def disambiguate_candidates(self, all_triples, all_definitions: List[List[dict]], all_query_encodings: List[List], all_definition_encodings: List[List]
                                ) -> list:

        all_mapped_triples = []
        for triples, definitions, query_encodings, definition_encodings in tqdm(zip(all_triples, all_definitions, all_query_encodings, all_definition_encodings), total=len(all_triples)):
            candidates = self.generate_candidates(query_encodings)
            if self.disambiguator_model is None:
                linked_entities = self.llm_disambiguate(definitions, candidates)
            else:
                linked_entities = self.ce_disambiguate(definitions, candidates)
            entity_mapping = {}
            for linked_entity, definition, definition_encoding in zip(linked_entities, definitions,
                                                                      definition_encodings):
                if linked_entity is None and definition["entity"] not in entity_mapping:
                    self.internal_entity_reps[definition["entity"]] = {"encodings": [definition_encoding],
                                                                       "identifier": f"http://createdbyedc.org/entities/{len(self.internal_entity_reps)}",
                                                                       "samples": [definition]}
                    entity_mapping[definition["entity"]] = self.internal_entity_reps[definition["entity"]]
                else:
                    if "samples" in linked_entity:
                        linked_entity["samples"].append(definition)
                        linked_entity["encodings"].append(definition_encoding)
                    entity_mapping[definition["entity"]] = linked_entity

            mapped_triples = []
            for s, p, o in triples:
                subject_entity = entity_mapping[str(s)]
                object_entity = entity_mapping[str(o)]
                mapped_triples.append((subject_entity, p, object_entity))
            all_mapped_triples.append(mapped_triples)

        entity_mapped_triples = []
        for idx in range(len(all_mapped_triples)):
            mapped_linked_triplet = []
            for triple in all_mapped_triples[idx]:
                s, p, o = triple
                if "identifier" in s:
                    new_s = s["identifier"]
                else:
                    continue
                if "identifier" in o:
                    new_o = o["identifier"]
                else:
                    continue
                mapped_linked_triplet.append((new_s, p, new_o))
            entity_mapped_triples.append(mapped_linked_triplet)

        return entity_mapped_triples


    @staticmethod
    def parse_output(output: str, entities: set) -> list:
        start_bracket = output.find("[")
        end_bracket = output.rfind("]")
        content = []
        if start_bracket != -1 and end_bracket != -1:
            try:
                relevant_text = output[start_bracket:end_bracket + 1]
                relevant_text = relevant_text.replace('\_', '_')
                content = json.loads(relevant_text)
            except json.JSONDecodeError as e:
                content = []
            if isinstance(content, list):
                content = [x for x in content if isinstance(x, dict)]

        final_entities = {}
        for entity in content:
            if entity.get("entity", "") in entities:
                if "type" not in entity:
                    entity["type"] = ""
                if "definition" not in entity:
                    entity["definition"] = ""
                final_entities[entity["entity"]] = entity
        incomplete = 0
        for entity in entities:
            if entity not in final_entities:
                final_entities[entity] = {"entity": entity, "definition": "", "type": ""}
                incomplete += 1

        return list(final_entities.values()), incomplete
    def generate_entity_descriptions(self, all_input_text_str: List[str], all_triples: List[List[list]]) -> list:
        all_messages = []
        all_entities = []
        for input_text_str, triples in zip(all_input_text_str, all_triples):
            entities = set()
            for triple in triples:
                entities.add(str(triple[0]))
                entities.add(str(triple[2]))
            entity_str = "\n".join(entities)
            filled_prompt = self.entity_description_prompt.format_map(
                {"few_shot_examples": self.entity_description_few_shot_examples,
                 "entities": entity_str}
            )
            user_prompt = f"Text: {input_text_str.strip()}\nEntities:\n{entity_str}"

            messages = [{"role": "system", "content": filled_prompt},
                        {"role": "user", "content": user_prompt}]
            all_messages.append(messages)
            all_entities.append(entities)
        completions = self.generate(all_messages)
        all_generated_definitions = []
        all_incomplete_parses = 0
        overall_entities = 0
        for completion, entities in zip(completions, all_entities):
            generated_definitions, incomplete_parses = self.parse_output(completion, entities)
            all_incomplete_parses += incomplete_parses
            overall_entities += len(entities)
            all_generated_definitions.append(generated_definitions)
        logger.info(f"Incomplete parses {all_incomplete_parses/overall_entities}")
        return all_generated_definitions

    def get_definition_encodings(self, all_entity_definitions, all_input_text):
        all_entity_strs  = []
        for entity_definitions, input_text in zip(all_entity_definitions, all_input_text):
            entity_strs = []
            for entity in entity_definitions:
                entity_strs.append(create_full_entity_description(
                    entity["entity"],
                    entity["definition"],
                    [entity["type"]],
                ))

            all_entity_strs += entity_strs
        if isinstance(self.sentence_encoder, LLM):
            all_query_entity_strs = [STS_PROMPT + x for x in all_entity_strs]
            all_query_encodings = self.sentence_encoder.embed(all_query_entity_strs)
            all_sentence_encodings = self.sentence_encoder.embed(all_entity_strs)
            all_query_encodings = np.array([x.outputs.embedding for x in all_query_encodings], dtype=np.float32)
            all_query_encodings = normalize(all_query_encodings, axis=1)
            all_sentence_encodings = np.array([x.outputs.embedding for x in all_sentence_encodings], dtype=np.float32)
            all_sentence_encodings = normalize(all_sentence_encodings, axis=1)
        else:
            all_query_entity_strs = [x for x in
                                     all_entity_strs]

            all_query_encodings = self.sentence_encoder.encode(all_query_entity_strs, normalize_embeddings=True).astype(np.float32)
            all_sentence_encodings = self.sentence_encoder.encode(all_entity_strs,
                                                               normalize_embeddings=True).astype(np.float32)
        final_query_encodings = []
        final_sentence_encodings = []
        offset = 0
        for entity_definitions in all_entity_definitions:
            query_encodings = all_query_encodings[offset:offset + len(entity_definitions)]
            sentence_encodings = all_sentence_encodings[offset:offset + len(entity_definitions)]
            offset += len(entity_definitions)
            final_query_encodings.append(query_encodings)
            final_sentence_encodings.append(sentence_encodings)

        return final_query_encodings, final_sentence_encodings


    def link_entities(self, all_input_text_str: List[str], all_triples: List[List[list]]):
        logger.info("Generating entity descriptions...")
        all_definitions = self.generate_entity_descriptions(all_input_text_str, all_triples)
        logger.info("Generating entity descriptions done.")
        all_query_encodings, all_definition_encodings = self.get_definition_encodings(all_definitions, all_input_text_str)
        logger.info("Generating entity encodings done.")
        if self.cluster:
            mention_entity_scores, mention_mention_scors = self.calculate_neighbors(all_definitions,
                                                                                    all_query_encodings,
                                                                                    all_definition_encodings)

            all_mapped_triples = self.disambiguate_by_clustering(all_triples, all_definitions, mention_entity_scores, mention_mention_scors)
        else:
            all_mapped_triples = self.disambiguate_candidates(all_triples, all_definitions, all_query_encodings, all_definition_encodings)
        return all_mapped_triples

