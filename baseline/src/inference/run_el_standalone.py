import json
import logging
import os
from argparse import ArgumentParser

import faiss
from sentence_transformers import SentenceTransformer, CrossEncoder
from vllm import LLM

import edc.utils.llm_utils as llm_utils
from edc.entity_extraction import EntityExtractor
from edc.entity_linking import EntityLinker

os.environ["TOKENIZERS_PARALLELISM"] = "false"

if __name__ == "__main__":
    parser = ArgumentParser(description="Standalone entity recognition and linking.")

    parser.add_argument(
        "input_text_file_path",
        type=str,
        help="Text file with one input text per line.",
    )
    parser.add_argument(
        "--output_file_path",
        type=str,
        default="./el_standalone.jsonl",
        help="Path to the output JSONL file.",
    )

    # Entity Extraction settings
    parser.add_argument(
        "--ee_llm",
        default="meta-llama/Llama-3.1-8B-Instruct",
        help="LLM used for entity extraction.",
    )
    parser.add_argument(
        "--ee_prompt_template_file_path",
        default="edc/prompt_templates/ee_template.txt",
        help="Prompt template used for entity extraction.",
    )
    parser.add_argument(
        "--ee_few_shot_example_file_path",
        default="edc/few_shot_examples/rebel/ee_few_shot_examples.txt",
        help="Few-shot examples used for entity extraction.",
    )

    # Entity Linking settings
    parser.add_argument(
        "--el_llm",
        default="meta-llama/Llama-3.1-8B-Instruct",
        help="LLM used for entity disambiguation.",
    )
    parser.add_argument(
        "--el_embedder",
        default="intfloat/e5-mistral-7b-instruct",
        help="Sentence encoder for FAISS candidate retrieval.",
    )
    parser.add_argument(
        "--el_index",
        default=None,
        help="Path to the FAISS entity index file.",
    )
    parser.add_argument(
        "--el_mapping",
        default=None,
        help="Path to the JSON index-to-entity mapping file.",
    )
    parser.add_argument(
        "--el_disambiguator",
        default=None,
        help="Path to CrossEncoder re-ranker model.",
    )
    parser.add_argument(
        "--el_adapter_path",
        default=None,
        help="LoRA adapter path for the EL embedder.",
    )
    parser.add_argument(
        "--el_prompt_path",
        default="data",
        help="Directory containing the four EL prompt .txt files.",
    )
    parser.add_argument(
        "--me_threshold",
        type=float,
        default=0.5,
        help="Mention-entity similarity threshold.",
    )
    parser.add_argument(
        "--mm_threshold",
        type=float,
        default=0.5,
        help="Mention-mention similarity threshold.",
    )
    parser.add_argument(
        "--path_threshold",
        type=float,
        default=0.75,
        help="Clustering path threshold.",
    )
    parser.add_argument(
        "--cluster",
        action="store_true",
        help="Use clustering-based disambiguation.",
    )
    parser.add_argument("--logging_verbose", action="store_const", dest="loglevel", const=logging.INFO)
    parser.add_argument("--logging_debug", action="store_const", dest="loglevel", const=logging.DEBUG)

    args = parser.parse_args()
    logging.basicConfig(level=args.loglevel)

    # --- Read input texts ---
    if not os.path.exists(args.input_text_file_path):
        raise FileNotFoundError(f"Input file not found: {args.input_text_file_path}")
    with open(args.input_text_file_path, encoding="utf-8") as f:
        input_text_list = [line.strip() for line in f if line.strip()]
    logging.info(f"Loaded {len(input_text_list)} input texts.")

    # --- Stage 1: Entity Extraction ---
    if not os.path.exists(args.ee_prompt_template_file_path):
        raise FileNotFoundError(f"EE prompt template not found: {args.ee_prompt_template_file_path}")
    if not os.path.exists(args.ee_few_shot_example_file_path):
        raise FileNotFoundError(f"EE few-shot examples not found: {args.ee_few_shot_example_file_path}")

    ee_template = open(args.ee_prompt_template_file_path).read()
    ee_few_shot = open(args.ee_few_shot_example_file_path).read()

    if llm_utils.is_model_openai(args.ee_llm):
        entity_extractor = EntityExtractor(openai_model=args.ee_llm)
    else:
        ee_model = LLM(model=args.ee_llm, gpu_memory_utilization=0.5, enable_lora=True, max_model_len=4096)
        ee_tokenizer = ee_model.llm_engine.tokenizer.tokenizer
        entity_extractor = EntityExtractor(model=ee_model, tokenizer=ee_tokenizer)

    logging.info("Running entity extraction...")
    all_extracted_entities = entity_extractor.extract_entities(input_text_list, ee_few_shot, ee_template)
    logging.info("Entity extraction done.")

    # --- Stage 2: Build dummy triples ---
    # Use [mention, mention, mention] so the predicate (not remapped by the linker)
    # can be used to recover the original mention string from each output triple.
    all_dummy_triples = [
        [[m, m, m] for m in mentions]
        for mentions in all_extracted_entities
    ]

    # --- Stage 3: Build EntityLinker (replicates EDC.prepare_entity_linking()) ---
    if llm_utils.is_model_openai(args.el_llm):
        el_model, el_tokenizer, openai_model = None, None, args.el_llm
    else:
        el_model = LLM(model=args.el_llm, gpu_memory_utilization=0.5, enable_lora=True, max_model_len=4096)
        el_tokenizer = el_model.llm_engine.tokenizer.tokenizer
        openai_model = None

    if args.el_index is None:
        el_index = None
        el_mapping = {}
    else:
        if not os.path.exists(args.el_index):
            raise FileNotFoundError(f"EL index not found: {args.el_index}")
        if not os.path.exists(args.el_mapping):
            raise FileNotFoundError(f"EL mapping not found: {args.el_mapping}")
        el_index = faiss.read_index(args.el_index)
        el_mapping = json.load(open(args.el_mapping, "r"))
        el_mapping = {int(k): v for k, v in el_mapping.items()}

    try:
        el_embedder = LLM(model=args.el_embedder, task="embed", enable_lora=True, gpu_memory_utilization=0.45)
    except Exception as e:
        logging.warning(f"Could not load {args.el_embedder} as vLLM embed model ({e}), falling back to SentenceTransformer.")
        el_embedder = SentenceTransformer(args.el_embedder, trust_remote_code=True).float()

    if args.el_adapter_path is not None:
        el_embedder = (el_embedder, args.el_adapter_path)

    el_disambiguator = CrossEncoder(args.el_disambiguator) if args.el_disambiguator is not None else None

    entity_linker = EntityLinker(
        sentence_encoder=el_embedder,
        entity_index=el_index,
        index_to_entity=el_mapping,
        internal_entities=[],
        model=el_model,
        tokenizer=el_tokenizer,
        openai_model=openai_model,
        disambiguator_model=el_disambiguator,
        cluster=args.cluster,
        me_threshold=args.me_threshold,
        mm_threshold=args.mm_threshold,
        path_threshold=args.path_threshold,
        prompt_path=args.el_prompt_path,
    )

    # --- Stage 4: Link entities ---
    logging.info("Running entity linking...")
    all_linked_triples = entity_linker.link_entities(input_text_list, all_dummy_triples)
    logging.info("Entity linking done.")

    # --- Stage 5: Map output triples back to mention → URI ---
    # The predicate in each output triple is the original mention string (predicates are
    # not remapped by the linker), so we use it as the lookup key.
    final_results = []
    for idx, (text, mentions, linked_triples) in enumerate(
        zip(input_text_list, all_extracted_entities, all_linked_triples)
    ):
        mention_to_uri = {m: None for m in mentions}
        for (s_uri, mention_str, _) in linked_triples:
            if mention_str in mention_to_uri:
                mention_to_uri[mention_str] = s_uri

        final_results.append({
            "index": idx,
            "input_text": text,
            "extracted_mentions": mentions,
            "linked_entities": [
                {"mention": m, "uri": mention_to_uri[m]}
                for m in mentions
            ],
        })

    # --- Stage 6: Write output ---
    output_dir = os.path.dirname(os.path.abspath(args.output_file_path))
    os.makedirs(output_dir, exist_ok=True)
    with open(args.output_file_path, "w", encoding="utf-8") as f:
        for record in final_results:
            f.write(json.dumps(record, ensure_ascii=False) + "\n")

    logging.info(f"Results written to {args.output_file_path}")
