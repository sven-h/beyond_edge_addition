from typing import List
import os
from pathlib import Path
import edc.utils.llm_utils as llm_utils
import re
from edc.utils.e5_mistral_utils import MistralForSequenceEmbedding
from transformers import AutoModelForCausalLM, AutoTokenizer
import numpy as np
import copy


class EntityExtractor:
    # The class to handle the last stage: Schema Canonicalization
    def __init__(self, model: AutoModelForCausalLM = None, tokenizer: AutoTokenizer = None, openai_model=None) -> None:
        # The canonicalizer uses an embedding model to first fetch candidates from the target schema, then uses a verifier schema to decide which one to canonicalize to or not
        # canonoicalize at all.

        assert openai_model is not None or (model is not None and tokenizer is not None)
        self.model = model
        self.tokenizer = tokenizer
        self.openai_model = openai_model

    def extract_entities(self, input_text_str_list: List[str], few_shot_examples_str: str, prompt_template_str: str):
        all_messages = []
        for input_text_str in input_text_str_list:
            filled_prompt = prompt_template_str.format_map(
                {"few_shot_examples": few_shot_examples_str, "input_text": input_text_str}
            )
            messages = [{"role": "user", "content": filled_prompt}]
            all_messages.append(messages)

        if self.openai_model is None:
            # llm_utils.generate_completion_transformers([messages], self.model, self.tokenizer, device=self.device)
            completions = llm_utils.generate_completion_transformers(
                all_messages, self.model, self.tokenizer, answer_prepend="Entities: "
            )
        else:
            completions = []
            for messages in all_messages:
                completion = llm_utils.openai_chat_completion(self.openai_model, None, messages)
                completions.append(completion)
        all_extracted_entities_list = []
        for completion in completions:
            extracted_entities = llm_utils.parse_raw_entities(completion)
            all_extracted_entities_list.append(extracted_entities)
        return all_extracted_entities_list

    def merge_entities(
        self, input_text_list: List[str], all_entity_list_1: List[List[str]], all_entity_list_2: List[List[str]], prompt_template_str: str
    ):
        all_messages = []
        for input_text, entity_list_1, entity_list_2 in zip(input_text_list, all_entity_list_1, all_entity_list_2):
            filled_prompt = prompt_template_str.format_map(
                {"input_text": input_text, "entity_list_1": entity_list_1, "entity_list_2": entity_list_2}
            )
            messages = [{"role": "user", "content": filled_prompt}]
            all_messages.append(messages)

        if self.openai_model is None:
            # llm_utils.generate_completion_transformers([messages], self.model, self.tokenizer, device=self.device)
            completions = llm_utils.generate_completion_transformers(
                all_messages, self.model, self.tokenizer, answer_prepend="Answer: "
            )
        else:
            completions = []
            for messages in all_messages:
                completion = llm_utils.openai_chat_completion(self.openai_model, None, messages)
                completions.append(completion)
        all_extracted_entities_list = []
        for completion, list_1, list_2 in zip(completions, all_entity_list_1, all_entity_list_2):
            extracted_entities = llm_utils.parse_raw_entities(completion)
            if len(extracted_entities) == 0:
                if len(list_1) == 0:
                    extracted_entities = list_2
                else:
                    extracted_entities = list_1
            all_extracted_entities_list.append(extracted_entities)
        return all_extracted_entities_list

    def retrieve_relevant_relations(self, query_input_text: str, top_k=10):
        target_relation_list = list(self.target_schema_embedding_dict.keys())
        target_relation_embedding_list = list(self.target_schema_embedding_dict.values())

        query_embedding = llm_utils.get_embedding_e5mistral(
            self.model,
            self.tokenizer,
            query_input_text,
            "Retrieve descriptions of relations that are present in the given text.",
        )
        scores = np.array([query_embedding]) @ np.array(target_relation_embedding_list).T

        scores = scores[0]
        highest_score_indices = np.argsort(-scores)

        return {
            target_relation_list[idx]: self.target_schema_dict[target_relation_list[idx]]
            for idx in highest_score_indices[:top_k]
        }, [scores[idx] for idx in highest_score_indices[:top_k]]
