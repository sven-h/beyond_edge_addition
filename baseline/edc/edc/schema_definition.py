from typing import List
import os
from pathlib import Path
import edc.utils.llm_utils as llm_utils
import re
from transformers import AutoModelForCausalLM, AutoTokenizer
import logging

logger = logging.getLogger(__name__)


class SchemaDefiner:
    # The class to handle the first stage: Open Information Extraction
    def __init__(self, model: AutoModelForCausalLM = None, tokenizer: AutoTokenizer = None, openai_model=None) -> None:
        assert openai_model is not None or (model is not None and tokenizer is not None)
        self.model = model
        self.tokenizer = tokenizer
        self.openai_model = openai_model

    def define_schema(
        self,
        all_input_text_str: List[str],
        all_extracted_triplets_list: List[List[str]],
        few_shot_examples_str: str,
        prompt_template_str: str,
    ) -> List[List[str]]:
        # Given a piece of text and a list of triplets extracted from it, define each of the relation present

        all_messages = []
        for input_text_str, extracted_triplets_list in zip(all_input_text_str, all_extracted_triplets_list):
            relations_present = set()
            for t in extracted_triplets_list:
                relations_present.add(t[1])

            filled_prompt = prompt_template_str.format_map(
                {
                    "text": input_text_str,
                    "few_shot_examples": few_shot_examples_str,
                    "relations": relations_present,
                    "triples": extracted_triplets_list,
                }
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
        all_relation_definition_dict = []
        for completion, relations_present in zip(completions, all_messages):
            relation_definition_dict = llm_utils.parse_relation_definition(completion)
            all_relation_definition_dict.append(relation_definition_dict)

        # missing_relations = [rel for rel in relations_present if rel not in relation_definition_dict]
        # if len(missing_relations) != 0:
        #     logger.debug(f"Relations {missing_relations} are missing from the relation definition!")
        return all_relation_definition_dict
