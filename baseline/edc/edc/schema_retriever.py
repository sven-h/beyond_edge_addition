from typing import List
import os
from pathlib import Path
import edc.utils.llm_utils as llm_utils
import re
from edc.utils.e5_mistral_utils import MistralForSequenceEmbedding
from transformers import AutoModelForCausalLM, AutoTokenizer
import numpy as np
import copy

from src.special_prompts import RE_RET_PROMPT


class SchemaRetriever:
    # The class to handle the last stage: Schema Canonicalization
    def __init__(
        self, target_schema_dict: dict, target_kg_schema: dict, embedding_model, embedding_tokenizer, finetuned_e5mistral=False
    ) -> None:
        # The canonicalizer uses an embedding model to first fetch candidates from the target schema, then uses a verifier schema to decide which one to canonicalize to or not
        # canonoicalize at all.

        self.target_schema_dict = target_schema_dict
        self.target_kg_schema = target_kg_schema
        self.embedding_model = embedding_model
        self.embedding_tokenizer = embedding_tokenizer

        # Embed the target schema

        self.target_schema_embedding_dict = {}
        self.target_kg_schema_embedding_dict = {}
        self.finetuned_e5mistral = finetuned_e5mistral

        if finetuned_e5mistral:
            for relation, relation_definition in target_schema_dict.items():
                embedding = llm_utils.get_embedding_e5mistral(
                    self.embedding_model,
                    self.embedding_tokenizer,
                    relation_definition,
                )
                self.target_schema_embedding_dict[relation] = embedding

        else:
            if len(target_schema_dict) > 0:
                embeddings = llm_utils.get_embedding_sts(
                    self.embedding_model,
                    [f"{relation}: {relation_definition}" for relation, relation_definition in target_schema_dict.items()],
                )
            else:
                embeddings = []
            for relation, embedding in zip(target_schema_dict.keys(), embeddings):
                self.target_schema_embedding_dict[relation] = embedding

        if target_kg_schema is not None:
            if finetuned_e5mistral:
                for relation, relation_definition in target_kg_schema.items():
                    embedding = llm_utils.get_embedding_e5mistral(
                        self.embedding_model,
                        self.embedding_tokenizer,
                        relation_definition,
                    )
                    self.target_kg_schema_embedding_dict[relation] = embedding
            else:
                embeddings = llm_utils.get_embedding_sts(
                    self.embedding_model,
                    [f"{relation}: {relation_definition}" for relation, relation_definition in target_kg_schema.items()],
                )
                for relation, embedding in zip(target_kg_schema.keys(), embeddings):
                    self.target_kg_schema_embedding_dict[relation] = embedding

    def retrieve_relevant_relations(self, query_input_texts: List[str], top_k=10):
        target_relation_list = list(self.target_schema_embedding_dict.keys())
        target_kg_relation_list = list(self.target_kg_schema_embedding_dict.keys())
        target_relation_embedding_list = list(self.target_schema_embedding_dict.values())
        target_kg_relation_embedding_list = list(self.target_kg_schema_embedding_dict.values())

        query_embeddings = []
        if self.finetuned_e5mistral:
            for query_input_text in query_input_texts:
                query_embedding = llm_utils.get_embedding_e5mistral(
                    self.embedding_model,
                    self.embedding_tokenizer,
                    query_input_text,
                    "Retrieve descriptions of relations that are present in the given text.",
                )
                query_embeddings.append(query_embedding)
            query_embeddings = np.array(query_embeddings)
        else:
            query_embeddings = llm_utils.get_embedding_sts(
                self.embedding_model,
                query_input_texts,
                prompt=RE_RET_PROMPT
            )

        if len(target_relation_embedding_list) > 0:
            scores = query_embeddings @ np.array(target_relation_embedding_list).T
        else:
            scores = np.zeros((len(query_input_texts), len(target_relation_list)))

        kg_scores = query_embeddings @ np.array(target_kg_relation_embedding_list).T

        outputs = []
        for scores_, kg_scores_ in zip(scores, kg_scores):
            highest_score_indices = np.argsort(-scores_).reshape(-1)
            highest_kg_score_indices = np.argsort(-kg_scores_).reshape(-1)

            outputs.append([target_relation_list[idx] for idx in highest_score_indices[:top_k]] + [target_kg_relation_list[idx] for idx in highest_kg_score_indices[:top_k]])
        return outputs
