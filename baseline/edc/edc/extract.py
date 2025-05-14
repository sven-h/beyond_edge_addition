from typing import List
import os
from pathlib import Path
import edc.utils.llm_utils as llm_utils
import re
from transformers import AutoModelForCausalLM, AutoTokenizer


class Extractor:
    # The class to handle the first stage: Open Information Extraction
    def __init__(self, model: AutoModelForCausalLM = None, tokenizer: AutoTokenizer = None, openai_model=None) -> None:
        assert openai_model is not None or (model is not None and tokenizer is not None)
        self.model = model
        self.tokenizer = tokenizer
        self.openai_model = openai_model

    def extract(
        self,
        input_text_str_list: List[str],
        few_shot_examples_str: str,
        prompt_template_str: str,
        entities_hint_list: List[str] = None,
        relations_hint_list: List[str] = None,
    ) -> List[List[str]]:
        assert (entities_hint_list is None and relations_hint_list is None) or (
            entities_hint_list is not None and relations_hint_list is not None
        )
        if entities_hint_list is None:
            entities_hint_list = [None] * len(input_text_str_list)
            relations_hint_list = [None] * len(input_text_str_list)
        all_messages = []
        for input_text_str, entities_hint, relations_hint in zip(
            input_text_str_list, entities_hint_list, relations_hint_list
        ):
            filled_prompt = prompt_template_str.format_map(
                {
                    "few_shot_examples": few_shot_examples_str,
                    "input_text": input_text_str,
                    "entities_hint": entities_hint,
                    "relations_hint": relations_hint,
                }
            )
            messages = [{"role": "user", "content": filled_prompt}]

            all_messages.append(messages)

        completions = []
        if self.openai_model is None:
            # llm_utils.generate_completion_transformers([messages], self.model, self.tokenizer, device=self.device)
            completions = llm_utils.generate_completion_transformers(
                all_messages, self.model, self.tokenizer, answer_prepend="Triplets: "
            )
        else:
            for messages in all_messages:
                completion = llm_utils.openai_chat_completion(self.openai_model, None, messages)
                completions.append(completion)
        all_extracted_triplets_list = []
        for completion in completions:
            extracted_triplets_list = llm_utils.parse_raw_triplets(completion)
            all_extracted_triplets_list.append(extracted_triplets_list)
        return all_extracted_triplets_list
