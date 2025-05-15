import json
from argparse import ArgumentParser
from edc.edc_framework import EDC
import os
import logging

from src.evaluate import load_dataset

os.environ["TOKENIZERS_PARALLELISM"] = "false"

if __name__ == "__main__":
    parser = ArgumentParser()

    parser.add_argument("data_path", type=str)
    # OIE module setting
    parser.add_argument(
        "--oie_llm", default="meta-llama/Llama-3.1-8B-Instruct", help="LLM used for open information extraction."
    )
    parser.add_argument(
        "--oie_prompt_template_file_path",
        default="edc/prompt_templates/oie_template.txt",
        help="Promp template used for open information extraction.",
    )
    parser.add_argument(
        "--oie_few_shot_example_file_path",
        default="edc/few_shot_examples/rebel/oie_few_shot_examples.txt",
        help="Few shot examples used for open information extraction.",
    )

    # Schema Definition setting
    parser.add_argument(
        "--sd_llm", default="meta-llama/Llama-3.1-8B-Instruct", help="LLM used for schema definition."
    )
    parser.add_argument(
        "--sd_prompt_template_file_path",
        default="edc/prompt_templates/sd_template.txt",
        help="Prompt template used for schema definition.",
    )
    parser.add_argument(
        "--sd_few_shot_example_file_path",
        default="edc/few_shot_examples/rebel/sd_few_shot_examples.txt",
        help="Few shot examples used for schema definition.",
    )

    # Schema Canonicalization setting
    parser.add_argument(
        "--sc_llm",
        default="meta-llama/Llama-3.1-8B-Instruct",
        help="LLM used for schema canonicaliztion verification.",
    )
    parser.add_argument(
        "--sc_embedder", default="intfloat/e5-mistral-7b-instruct",
        help="Embedder used for schema canonicalization. Has to be a sentence transformer. Please refer to https://sbert.net/"
    )
    parser.add_argument(
        "--sc_prompt_template_file_path",
        default="edc/prompt_templates/sc_template.txt",
        help="Prompt template used for schema canonicalization verification.",
    )

    # Refinement setting
    parser.add_argument("--sr_adapter_path", default=None, help="Path to adapter of schema retriever.")
    parser.add_argument(
        "--sr_embedder", default="intfloat/e5-mistral-7b-instruct",
        help="Embedding model used for schema retriever. Has to be a sentence transformer. Please refer to https://sbert.net/"
    )
    parser.add_argument(
        "--oie_refine_prompt_template_file_path",
        default="edc/prompt_templates/oie_r_template.txt",
        help="Prompt template used for refined open information extraction.",
    )
    parser.add_argument(
        "--oie_refine_few_shot_example_file_path",
        default="edc/few_shot_examples/rebel/oie_few_shot_refine_examples.txt",
        help="Few shot examples used for refined open information extraction.",
    )
    parser.add_argument(
        "--ee_llm", default="meta-llama/Llama-3.1-8B-Instruct", help="LLM used for entity extraction."
    )
    parser.add_argument(
        "--ee_prompt_template_file_path",
        default="edc/prompt_templates/ee_template.txt",
        help="Prompt templated used for entity extraction.",
    )
    parser.add_argument(
        "--ee_few_shot_example_file_path",
        default="edc/few_shot_examples/rebel/ee_few_shot_examples.txt",
        help="Few shot examples used for entity extraction.",
    )
    parser.add_argument(
        "--em_prompt_template_file_path",
        default="edc/prompt_templates/em_template.txt",
        help="Prompt template used for entity merging.",
    )

    parser.add_argument("--me_threshold", default=0.5, type=float, help="Threshold for entity merging.")
    parser.add_argument("--mm_threshold", default=0.5, type=float, help="Threshold for entity merging.")
    parser.add_argument("--path_threshold", default=0.75, type=float, help="Threshold for entity merging.")


    # Input setting
    parser.add_argument(
        "--input_text_file_path",
        default="./datasets/example.txt",
        help="File containing input texts to extract KG from, each line contains one piece of text.",
    )
    parser.add_argument(
        "--target_schema_path",
        default="../relation_schema.csv",
        help="File containing the target schema to align to.",
    )
    parser.add_argument("--refinement_iterations", default=0, type=int, help="Number of iteration to run.")
    parser.add_argument(
        "--enrich_schema",
        action="store_true",
        help="Whether un-canonicalizable relations should be added to the schema.",
    )

    parser.add_argument(
        "--el_llm",
        default="meta-llama/Llama-3.1-8B-Instruct",
        help="LLM used for entity linking.",
    )

    parser.add_argument("--cluster",  action="store_true", help="Whether to use cluster mode.")
    parser.add_argument("--initial_refine", action="store_true")
    parser.add_argument("--block_refine_relations", action="store_true")

    parser.add_argument(
        "--el_index",
        default=None,
        help="Path to the entity linking index.",
    )

    parser.add_argument(
        "--include_relation_example", type=str, default="self")

    parser.add_argument(
        "--el_mapping",
        default=None,
        help="Path to the entity linking mapping.",
    )

    parser.add_argument(
        "--el_disambiguator",
        default=None,
        help="Path to the entity linking disambiguator.",
    )

    parser.add_argument(
        "--el_embedder",
        default="intfloat/e5-mistral-7b-instruct",
        help="Embedder",
    )
    parser.add_argument(
        "--el_adapter_path",
        default=None,
    )


    # Output setting
    parser.add_argument("--logging_verbose", action="store_const", dest="loglevel", const=logging.INFO)
    parser.add_argument("--logging_debug", action="store_const", dest="loglevel", const=logging.DEBUG)

    args = parser.parse_args()
    args = vars(args)
    edc = EDC(**args)

    data_path = args["data_path"]
    # Find folder of name iter{n} with highest n
    max_iter = -1
    for folder in os.listdir(data_path):
        if folder.startswith("iter") and folder[4:].isdigit():
            max_iter = max(max_iter, int(folder[4:]))

    schema_dict_path = os.path.join(data_path, f"iter{max_iter}", "schema_dict.json")
    result_at_stage_path = os.path.join(data_path, f"iter{max_iter}", "result_at_each_stage.json")

    schema_dict = json.load(open(schema_dict_path, "r"))
    result_at_stage = json.load(open(result_at_stage_path, "r"))
    can_triplets_list = []
    input_text_list = []
    for item in result_at_stage:
        can_triplets_list.append(item["schema_canonicalizaiton"])
        input_text_list.append(item["input_text"])

    linked_triplets_list = edc.do_linking(
        input_text_list,
        can_triplets_list,
        schema_dict
    )

    final_results = []
    for idx in range(len(linked_triplets_list)):
        mapped_linked_triplet = linked_triplets_list[idx]
        final_results.append({
            "index": idx,
            "input_text": input_text_list[idx],
            "linked_triplets": mapped_linked_triplet,
        })

    json.dump(
        final_results,
        open(os.path.join(data_path, f"iter{max_iter}", "linked_triplets.json"), "w"),
        ensure_ascii=False,
        indent=4,)

