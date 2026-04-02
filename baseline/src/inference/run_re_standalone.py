"""Standalone relation extraction from GT entity lists.

Input JSONL format (one record per line):
    {"text": "...", "entities": ["entity_label_1", "entity_label_2", ...]}

Pipeline:
    1. For each text, show all entities to an LLM and ask it to output all triples.
    2. Parse the output into (e1, relation_desc, e2) triples.
    3. Feed triples through EDC's Schema Definition and Schema Canonicalization stages.
    4. Write canonicalized triples to output JSONL.

Output JSONL format:
    {"index": 0, "input_text": "...", "canonicalized_triplets": [["e1", "relation_label", "e2"], ...]}
"""

import json
import logging
import os
import re
from argparse import ArgumentParser

from vllm import LLM

import edc.utils.llm_utils as llm_utils
from edc.edc_framework import EDC

os.environ["TOKENIZERS_PARALLELISM"] = "false"


def parse_triples(completion: str, valid_entities: set) -> list:
    """Parse LLM output into a list of [e1, relation, e2] triples.

    Expects lines of the form: (Entity 1, relation, Entity 2)
    Silently drops malformed lines and triples where e1 or e2 are not in valid_entities.
    """
    if completion.strip().lower() == "none":
        return []

    triples = []
    for line in completion.strip().splitlines():
        line = line.strip()
        if not line:
            continue
        # Match (anything, anything, anything) — allow commas inside fields
        m = re.match(r"^\((.+),(.+),(.+)\)$", line)
        if not m:
            continue
        e1, relation, e2 = m.group(1).strip(), m.group(2).strip(), m.group(3).strip()
        if e1 in valid_entities and e2 in valid_entities and e1 != e2 and relation:
            triples.append([e1, relation, e2])
    return triples


def extract_relations(
    input_text_list: list,
    all_entity_lists: list,
    re_model,
    re_tokenizer,
    re_template: str,
    re_few_shot: str,
) -> list:
    """Call the LLM once per text with all entities, return parsed triples per text."""
    all_messages = []
    for text, entities in zip(input_text_list, all_entity_lists):
        entities_str = str(entities)
        filled = re_template.format(
            few_shot_examples=re_few_shot,
            text=text,
            entities=entities_str,
        )
        all_messages.append([{"role": "user", "content": filled}])

    completions = llm_utils.generate_completion_transformers(
        all_messages, re_model, re_tokenizer
    )

    results = []
    for completion, entities in zip(completions, all_entity_lists):
        triples = parse_triples(completion, set(entities))
        results.append(triples)
        logging.debug(f"Parsed {len(triples)} triples from completion.")
    return results


if __name__ == "__main__":
    parser = ArgumentParser(description="Standalone relation extraction from GT entity lists.")

    parser.add_argument(
        "input_file_path",
        type=str,
        help="JSONL file with {text, entities} records.",
    )
    parser.add_argument(
        "--output_file_path",
        type=str,
        default="./re_standalone.jsonl",
        help="Path to the output JSONL file.",
    )

    # Directed RE LLM settings
    parser.add_argument(
        "--re_llm",
        default="meta-llama/Llama-3.1-8B-Instruct",
        help="LLM used for relation extraction.",
    )
    parser.add_argument(
        "--re_prompt_template_file_path",
        default="edc/prompt_templates/re_directed_template.txt",
        help="Prompt template for RE ({text}, {entities}, {few_shot_examples}).",
    )
    parser.add_argument(
        "--re_few_shot_example_file_path",
        default="edc/few_shot_examples/rebel/re_directed_few_shot_examples.txt",
        help="Few-shot examples for RE.",
    )

    # Schema Definition settings
    parser.add_argument("--sd_llm", default="meta-llama/Llama-3.1-8B-Instruct")
    parser.add_argument("--sd_prompt_template_file_path", default="edc/prompt_templates/sd_template.txt")
    parser.add_argument("--sd_few_shot_example_file_path", default="edc/few_shot_examples/rebel/sd_few_shot_examples.txt")

    # Schema Canonicalization settings
    parser.add_argument("--sc_llm", default="meta-llama/Llama-3.1-8B-Instruct")
    parser.add_argument("--sc_embedder", default="intfloat/e5-mistral-7b-instruct")
    parser.add_argument("--sc_prompt_template_file_path", default="edc/prompt_templates/sc_template.txt")
    parser.add_argument(
        "--target_schema_path",
        default=None,
        help="CSV file with the target relation schema to align to.",
    )
    parser.add_argument("--enrich_schema", action="store_true")

    parser.add_argument("--logging_verbose", action="store_const", dest="loglevel", const=logging.INFO)
    parser.add_argument("--logging_debug", action="store_const", dest="loglevel", const=logging.DEBUG)

    args = parser.parse_args()
    logging.basicConfig(level=args.loglevel)

    # --- Load input ---
    if not os.path.exists(args.input_file_path):
        raise FileNotFoundError(f"Input file not found: {args.input_file_path}")
    records = []
    with open(args.input_file_path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                records.append(json.loads(line))
    logging.info(f"Loaded {len(records)} records.")

    input_text_list = [r["text"] for r in records]
    all_entity_lists = [r["entities"] for r in records]

    # --- Stage 1: Relation Extraction ---
    for path, label in [
        (args.re_prompt_template_file_path, "RE prompt template"),
        (args.re_few_shot_example_file_path, "RE few-shot examples"),
    ]:
        if not os.path.exists(path):
            raise FileNotFoundError(f"{label} not found: {path}")

    re_template = open(args.re_prompt_template_file_path).read()
    re_few_shot = open(args.re_few_shot_example_file_path).read()

    if llm_utils.is_model_openai(args.re_llm):
        raise NotImplementedError("OpenAI RE model not yet supported in standalone mode.")
    re_model = LLM(model=args.re_llm, gpu_memory_utilization=0.5, enable_lora=True, max_model_len=4096)
    re_tokenizer = re_model.llm_engine.tokenizer.tokenizer

    logging.info("Running relation extraction...")
    re_triplets_list = extract_relations(
        input_text_list, all_entity_lists, re_model, re_tokenizer, re_template, re_few_shot
    )
    total = sum(len(t) for t in re_triplets_list)
    logging.info(f"RE done. Total triples extracted: {total}")

    # --- Stage 2 & 3: Schema Definition + Canonicalization via EDC ---
    edc_config = {
        "sd_llm": args.sd_llm,
        "sd_prompt_template_file_path": args.sd_prompt_template_file_path,
        "sd_few_shot_example_file_path": args.sd_few_shot_example_file_path,
        "sc_llm": args.sc_llm,
        "sc_embedder": args.sc_embedder,
        "sc_prompt_template_file_path": args.sc_prompt_template_file_path,
        "target_schema_path": args.target_schema_path,
        "enrich_schema": args.enrich_schema,
        # Required EDC keys for unused modules (not called in this script)
        "oie_llm": "meta-llama/Llama-3.1-8B-Instruct",
        "oie_prompt_template_file_path": "edc/prompt_templates/oie_template.txt",
        "oie_few_shot_example_file_path": "edc/few_shot_examples/rebel/oie_few_shot_examples.txt",
        "oie_refine_prompt_template_file_path": "edc/prompt_templates/oie_r_template.txt",
        "oie_refine_few_shot_example_file_path": "edc/few_shot_examples/rebel/oie_few_shot_refine_examples.txt",
        "ee_llm": "meta-llama/Llama-3.1-8B-Instruct",
        "ee_prompt_template_file_path": "edc/prompt_templates/ee_template.txt",
        "ee_few_shot_example_file_path": "edc/few_shot_examples/rebel/ee_few_shot_examples.txt",
        "em_prompt_template_file_path": "edc/prompt_templates/em_template.txt",
        "sr_adapter_path": None,
        "sr_embedder": "intfloat/e5-mistral-7b-instruct",
        "initial_refine": False,
        "block_refine_relations": False,
        "include_relation_example": "self",
        "skip_el": True,
        "loglevel": args.loglevel,
    }
    edc = EDC(**edc_config)

    logging.info("Running Schema Definition...")
    sd_dict_list = edc.schema_definition(input_text_list, re_triplets_list)

    logging.info("Running Schema Canonicalization...")
    empty_hints = [[] for _ in input_text_list]
    canon_triplets_list, _, _ = edc.schema_canonicalization(
        input_text_list,
        re_triplets_list,
        sd_dict_list,
        empty_hints,
    )

    # --- Stage 4: Write output ---
    final_results = [
        {
            "index": idx,
            "input_text": text,
            "canonicalized_triplets": [t for t in canon_triplets if t is not None],
        }
        for idx, (text, canon_triplets) in enumerate(zip(input_text_list, canon_triplets_list))
    ]

    output_dir = os.path.dirname(os.path.abspath(args.output_file_path))
    os.makedirs(output_dir, exist_ok=True)
    with open(args.output_file_path, "w", encoding="utf-8") as f:
        for record in final_results:
            f.write(json.dumps(record, ensure_ascii=False) + "\n")

    logging.info(f"Results written to {args.output_file_path}")
