# Beyond Edge Addition: A Dataset for Information Extraction Incorporating New Instances, Types, and Relations

## Abstract
Information extraction (IE) converts natural language text into structured triples of subject, predicate, and object. Existing IE datasets assume all entities (instances, classes) and relations are defined within a knowledge graph (KG), focusing on discovering the relationships between entities. However, this does not reflect real-world scenarios, where many entities and relations may be absent from the KG. Moreover, most datasets lack a snapshot of the corresponding knowledge graph, causing evaluation inconsistencies as different systems may use varying KG versions.
Such inconsistencies undermine fair benchmarking and reproducibility.
This paper introduces a novel information extraction dataset
specifically designed to better reflect realistic KG incompleteness.
Our dataset includes 20\% missing classes and instances, along with 5\% missing relations,
requiring systems to not only add new links (edges) but also propose new instances, classes, and relations.
To ensure reproducibility and prevent leakage from pre-trained language models,
we provide a heavily modified version of Wikidata where background knowledge cannot be exploited.
We present a baseline that employs a mix of large language and encoder-only models for extraction and disambiguation tasks. It operates in several iterations, initially identifying a first set of triples, then progressively refining them by referencing the KG. Finally, it links all entities either to the KG or identifies them as new entities using clustering. The results highlight the difficulties posed by the dataset, especially in recognizing the new entities and relations. 

## Repository Structure

All the code for the baseline is contained in the `baseline` folder.
The dataset and knowledge graph generation code is in the jupyter notebook contained in the `kg_and_dataset_generation` folder.

The actual dataset and produced knowledge graph can be found in [zenodo](https://doi.org/10.5281/zenodo.15398296). The password for unzipping the files is `nWwBQ_B`.

The dataset is also available at [Huggingface](https://huggingface.co/datasets/sven-h/beyond_edge_addition).