# Baseline
This directory contains the baseline model.
It relies on code from the NASTyLinker method (https://github.com/nheist/CaLiGraph) as well as code from the EDC framework (https://github.com/clear-nus/edc).

## Preparation
Several steps are necessary to get a running method. 
Given the dataset, the following steps are necessary:
1. Train the schema retriever using [create_schema_gen_dataset.py](src%2Fcreate_schema_gen_dataset.py) and [train_sentence_transformers.py](src%2Ftrain_sentence_transformers.py).
````
python create_schema_gen_dataset.py train.json --development_data_path dev.json --add_special_prompt
````
````
python train_sentence_transformers.py schema_gen_dataset schema_retriever
````
2. Generate entity mention definitions using [generate_mention_definitions.py](src%2Fgenerate_mention_definitions.py)

````
python generate_mention_definitions.py train.jsonl generated.jsonl
````

3. Train the candidate retrieval using [create_candidate_gen_dataset.py](src%2Fcreate_candidate_gen_dataset.py) and [train_sentence_transformers.py](src%2Ftrain_sentence_transformers.py)
````
python create_candidate_gen_dataset.py train_generated.jsonl --development_data_path dev_generated.jsonl
````
````
python train_sentence_transformers.py candidate_retrieval_dataset candidate_retriever
````
4. Create an candidate index using [construct_entity_index.py](src%2Fconstruct_entity_index.py)
````
python construct_entity_index.py --el_embedder_name candidate_retriever/final --index_name entity_index
````
5. Train the cross-encoder using [train_ce_entity_linking_dataset.py](src%2Ftrain_ce_entity_linking_dataset.py) by supplying it with the created candidate index.
````
python train_ce_entity_linking_dataset.py train_generated.jsonl dev_generated.jsonl --candidate_retrieval_model candidate_retriever/final --entity_index entity_index.index --entity_mapping entity_index.json
````

# Running the method 
Given all the generated files, indexes and models, the method can then be run on a list of texts using:
[run.py](edc%2Frun.py).
The method has several parameters that can be set via command line arguments. To reproduce the paper results, please set:
- `--cluster` (flag)
- `--initial_refine` (flag)
- `--refinement_iterations` (2)
- `--enrich_schema` (flag)

If the relation canonicalization of new relations shall be disabled, do not set `--enrich_schema`. 

| Argument | Default                                                        | Description                                                             |
|---------|----------------------------------------------------------------|-------------------------------------------------------------------------|
| `--oie_llm` | `meta-llama/Llama-3.1-8B-Instruct`                             | LLM used for open information extraction.                               |
| `--oie_prompt_template_file_path` | `edc/prompt_templates/oie_template.txt`                        | Prompt template used for open information extraction.                   |
| `--oie_few_shot_example_file_path` | `edc/few_shot_examples/rebel/oie_few_shot_examples.txt`        | Few-shot examples for open information extraction.                      |
| `--include_relation_example` | `self`                                                         | Whether to include relation examples in the prompt.                     |
| `--cluster` | *(flag)*                                                       | Enable clustering for entity linking (NASTyLinker).                     |
| `--initial_refine` | *(flag)*                                                       | Enable refinement in the first iteration.                               |
| `--block_refine_relations` | *(flag)*                                                       | Do not use refinement hints for the canonicalization.                   |
| `--sd_llm` | `meta-llama/Llama-3.1-8B-Instruct`                             | LLM used for schema definition.                                         |
| `--sd_prompt_template_file_path` | `edc/prompt_templates/sd_template.txt`                         | Prompt template for schema definition.                                  |
| `--sd_few_shot_example_file_path` | `edc/few_shot_examples/rebel/sd_few_shot_examples.txt`         | Few-shot examples for schema definition.                                |
| `--sc_llm` | `meta-llama/Llama-3.1-8B-Instruct`                             | LLM used for schema canonicalization verification.                      |
| `--sc_embedder` | `intfloat/e5-mistral-7b-instruct`                              | Embedder for schema canonicalization (must be a sentence transformer).  |
| `--sc_prompt_template_file_path` | `edc/prompt_templates/sc_template.txt`                         | Prompt template for schema canonicalization verification.               |
| `--sr_adapter_path` | `None`                                                         | Path to schema retriever adapter.                                       |
| `--sr_embedder` | `intfloat/e5-mistral-7b-instruct`                              | Embedding model for schema retriever.                                   |
| `--oie_refine_prompt_template_file_path` | `edc/prompt_templates/oie_r_template.txt`                      | Prompt template for refined OIE.                                        |
| `--oie_refine_few_shot_example_file_path` | `edc/few_shot_examples/rebel/oie_few_shot_refine_examples.txt` | Few-shot examples for refined OIE.                                      |
| `--ee_llm` | `meta-llama/Llama-3.1-8B-Instruct`                             | LLM used for entity extraction.                                         |
| `--ee_prompt_template_file_path` | `edc/prompt_templates/ee_template.txt`                         | Prompt template for entity extraction.                                  |
| `--ee_few_shot_example_file_path` | `edc/few_shot_examples/rebel/ee_few_shot_examples.txt`         | Few-shot examples for entity extraction.                                |
| `--em_prompt_template_file_path` | `edc/prompt_templates/em_template.txt`                         | Prompt template for entity merging.                                     |
| `--input_text_file_path` | `./datasets/example.txt`                                       | File containing input texts for KG extraction (one per line).           |
| `--target_schema_path` | `../relation_schema.csv`                                       | Path to target schema to align to.                                      |
| `--refinement_iterations` | `0`                                                            | Number of refinement iterations.                                        |
| `--enrich_schema` | *(flag)*                                                       | Add un-canonicalizable relations to the schema.                         |
| `--el_llm` | `meta-llama/Llama-3.1-8B-Instruct`                             | LLM used for entity linking.                                            |
| `--el_index` | `None`                                                         | Path to the entity linking index.                                       |
| `--el_mapping` | `None`                                                         | Path to the entity linking mapping.                                     |
| `--me_threshold` | `0.2`                                                          | Threshold for entity merging (me).                                      |
| `--mm_threshold` | `0.9`                                                          | Threshold for entity merging (mm).                                      |
| `--path_threshold` | `0.5`                                                          | Threshold for entity merging via path.                                  |
| `--el_disambiguator` | `None`                                                         | Path to the trained entity linking disambiguator (cross encoder).       |
| `--el_embedder` | `intfloat/e5-mistral-7b-instruct`                              | Embedder used for entity linking. Should be set with the trained model. |
| `--output_dir` | `./output/tmp`                                                 | Output directory.                                                       |
| `--logging_verbose` | *(flag)*                                                       | Set logging level to INFO.                                              |
| `--logging_debug` | *(flag)*                                                       | Set logging level to DEBUG.                                             |
