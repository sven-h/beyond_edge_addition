# Baseline
This directory contains the baseline model.
It relies on code from the NASTyLinker method (https://github.com/nheist/CaLiGraph) as well as code from the EDC framework (https://github.com/clear-nus/edc).

For reproducing the results we also provide the trained models at: https://zenodo.org/records/17828636?token=eyJhbGciOiJIUzUxMiJ9.eyJpZCI6IjFiODg2YTk2LWJjNjktNDkwZS1iMjkwLTA1NjVkZThhZWJhNSIsImRhdGEiOnt9LCJyYW5kb20iOiJmNTBmZDUzMDRiZjdjMjEzNGJkZDA1OWRlMTNiN2M4OSJ9.APtFa6LtvzdSCzrenfGWrRwQgBNA3vNmR-whULDzCI9dii8suOQ1jWJQhzTGg0TqgQIn-mUuhbl7JKfgxwP_Ag

## Preparation
Several steps are necessary to get a running method. 
Given the dataset, put the extracted files into the data directory. 
Then the following steps are necessary:
1. Train the schema retriever using [create_schema_gen_dataset.py](src%2Fpreparation%2Fcreate_schema_gen_dataset.py) and [train_sentence_transformers.py](src%2Ftraining%2Ftrain_sentence_transformers.py).
````
python create_schema_gen_dataset.py train.jsonl --development_data_path dev.jsonl --add_special_prompt
````
````
python train_sentence_transformers.py schema_gen_dataset schema_retriever
````
2. Generate entity mention definitions using [generate_mention_definitions.py](src%2Fpreparation%2Fgenerate_mention_definitions.py)

````
python generate_mention_definitions.py train.jsonl train_generated.jsonl
python generate_mention_definitions.py dev.jsonl dev_generated.jsonl
````

3. Train the candidate retrieval using [create_candidate_gen_dataset.py](src%2Fpreparation%2Fcreate_candidate_gen_dataset.py) and [train_sentence_transformers.py](src%2Ftraining%2Ftrain_sentence_transformers.py)
````
python create_candidate_gen_dataset.py train_generated.jsonl --development_data_path dev_generated.jsonl
````
````
python train_sentence_transformers.py candidate_retrieval_dataset candidate_retriever
````
4. Create a candidate index using [construct_entity_index.py](src%2Fpreparation%2Fconstruct_entity_index.py)
````
python construct_entity_index.py --el_embedder_name candidate_retriever/final --index_name entity_index
````
5. Train the cross-encoder using [train_ce_entity_linking_dataset.py](src%2Ftraining%2Ftrain_ce_entity_linking_dataset.py) by supplying it with the created candidate index.
````
python train_ce_entity_linking_dataset.py train_generated.jsonl dev_generated.jsonl crossencoder --candidate_retrieval_model candidate_retriever/final --entity_index entity_index.index --entity_mapping entity_index.json
````

# Running the method 
For evaluation the EDC framework expects several files to be present.
These can be generated using [prepare_edc_run.py](src%2Fpreparation%2Fprepare_edc_run.py). A GPU is necessary to run this script:
````
python prepare_edc_run.py --test_data_input_path test_dataset.jsonl --kg_data_path data --prompts_path data
````


Given all the generated files, indexes and models, the method can then be run on a list of texts using:
[run.py](edc%2Frun.py).
The method has several parameters that can be set via command line arguments. To reproduce the paper results, please set:
- `--cluster` (flag)
- `--initial_refine` (flag)
- `--refinement_iterations` (2)
- `--enrich_schema` (flag)

Furthermore, provide all the previously trained model checkpoints and indices via:
- `--sr_embedder` 
- `--el_index`
- `--el_mapping` 
- `--el_embedder` 
- `--el_disambiguator`

If the relation canonicalization of new relations shall be disabled, do not set `--enrich_schema`. 


## General
| Argument | Default                                                        | Description                                                             |
|---------|----------------------------------------------------------------|-------------------------------------------------------------------------|
| `--output_dir` | `./output/tmp`                                                 | Output directory.                                                       |
| `--logging_verbose` | *(flag)*                                                       | Set logging level to INFO.                                              |
| `--logging_debug` | *(flag)*                                                       | Set logging level to DEBUG.                                             |
| `--input_text_file_path` | `./datasets/example.txt`                                       | File containing input texts for KG extraction (one per line).           |
| `--target_schema_path` | `../relation_schema.csv`                                       | Path to target schema to align to.                                      |


## Prompts
| Argument | Default                                                        | Description                                                             |
|---------|----------------------------------------------------------------|-------------------------------------------------------------------------|
| `--oie_prompt_template_file_path` | `edc/prompt_templates/oie_template.txt`                        | Prompt template used for open information extraction.                   |
| `--oie_few_shot_example_file_path` | `edc/few_shot_examples/rebel/oie_few_shot_examples.txt`        | Few-shot examples for open information extraction.                      |
| `--sd_prompt_template_file_path` | `edc/prompt_templates/sd_template.txt`                         | Prompt template for schema definition.                                  |
| `--sd_few_shot_example_file_path` | `edc/few_shot_examples/rebel/sd_few_shot_examples.txt`         | Few-shot examples for schema definition.                                |
| `--sc_prompt_template_file_path` | `edc/prompt_templates/sc_template.txt`                         | Prompt template for schema canonicalization verification.               |
| `--oie_refine_prompt_template_file_path` | `edc/prompt_templates/oie_r_template.txt`                      | Prompt template for refined OIE.                                        |
| `--oie_refine_few_shot_example_file_path` | `edc/few_shot_examples/rebel/oie_few_shot_refine_examples.txt` | Few-shot examples for refined OIE.                                      |
| `--ee_prompt_template_file_path` | `edc/prompt_templates/ee_template.txt`                         | Prompt template for entity extraction.                                  |
| `--ee_few_shot_example_file_path` | `edc/few_shot_examples/rebel/ee_few_shot_examples.txt`         | Few-shot examples for entity extraction.                                |
| `--em_prompt_template_file_path` | `edc/prompt_templates/em_template.txt`                         | Prompt template for entity merging.                                     |


## Method Settings
| Argument | Default                            | Description                                                             |
|---------|------------------------------------|-------------------------------------------------------------------------|
| `--oie_llm` | `meta-llama/Llama-3.1-8B-Instruct` | LLM used for open information extraction.                               |
| `--include_relation_example` | `self`                             | Whether to include relation examples in the prompt.                     |
| `--initial_refine` | *(flag)*                           | Enable refinement in the first iteration.                               |
| `--block_refine_relations` | *(flag)*                           | Do not use refinement hints for the canonicalization.                   |
| `--sd_llm` | `meta-llama/Llama-3.1-8B-Instruct` | LLM used for schema definition.                                         |
| `--sc_llm` | `meta-llama/Llama-3.1-8B-Instruct` | LLM used for schema canonicalization verification.                      |
| `--sc_embedder` | `intfloat/e5-mistral-7b-instruct`  | Embedder for schema canonicalization (must be a sentence transformer).  |
| `--sr_adapter_path` | `None`                             | Path to schema retriever adapter.                                       |
| `--sr_embedder` | `intfloat/e5-mistral-7b-instruct`  | Embedding model for schema retriever.                                   |
| `--ee_llm` | `meta-llama/Llama-3.1-8B-Instruct` | LLM used for entity extraction.                                         |
| `--refinement_iterations` | `0`                                | Number of refinement iterations.                                        |
| `--enrich_schema` | *(flag)*                           | Add un-canonicalizable relations to the schema.                         |
| `--cluster` | *(flag)*                           | Enable clustering for entity linking (NASTyLinker).                     |
| `--el_disambiguator` | `None`                             | Path to the trained entity linking disambiguator (cross encoder).       |
| `--me_threshold` | `0.1`                              | Threshold for entity merging (me).                                      |
| `--mm_threshold` | `0.9`                              | Threshold for entity merging (mm).                                      |
| `--path_threshold` | `0.3`                              | Threshold for entity merging via path.                                  |
| `--el_embedder` | `intfloat/e5-mistral-7b-instruct`  | Embedder used for entity linking. Should be set with the trained model. |
| `--el_llm` | `meta-llama/Llama-3.1-8B-Instruct` | LLM used for entity linking.                                            |
| `--el_index` | `None`                             | Path to the entity linking index.                                       |
| `--el_mapping` | `None`                             | Path to the entity linking mapping.                                     |
