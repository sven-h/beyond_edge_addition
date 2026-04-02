# Beyond Edge Addition: A Dataset for Information Extraction Incorporating New Instances, Types, and Relations

## Overview
Traditional Information Extraction (IE) benchmarks operate under the assumption that all entities and relations are already defined within a target Knowledge Graph (KG). However, real-world KGs are dynamic and constantly evolve with new instances, classes, and relation types.

This work introduces a novel IE dataset designed to address this critical gap. Our benchmark specifically challenges systems to perform Open Information Extraction by not only adding new links (edges) but also correctly identifying and integrating novel entities, classes, and relations that are absent from the provided initial KG snapshot.

This dataset provides a more realistic and robust evaluation for IE systems operating in scenarios closer to real-world KG construction and maintenance.

## Repository Structure

All the code for the baseline is contained in the `baseline` folder.
The dataset and knowledge graph generation code is in the jupyter notebook contained in the `kg_and_dataset_generation` folder.

The actual dataset and produced knowledge graph can be found in [zenodo](https://doi.org/10.5281/zenodo.15398296). The password for unzipping the files is `nWwBQ_B`.

## Dataset Details

The dataset is an adaptation and significant modification of the high-quality SynthIE dataset, filtered and transformed to adhere to Description Logic (DL) constraints.

### Dataset Splits

The dataset is partitioned into standard splits for training, validation, and testing:

| Split      | Examples |
|------------|-------------------|
| Train      |  743,701          |
| Validation |  4,115            |
| Test       |  20,382           |


### Dataset Example Format

Each entry in the dataset is a JSON object representing a text snippet and the target facts (triples) that must be extracted.
Here is a simplified example showing how the ground truth is structured.
The following shows an example where only one triple needs to be extracted.
The object of the triple `Benoît Minisini` is not in the KG and needs to be created.
In the ground truth this is represented as a blank node. Additional information is provided for this missing entity.
Please note that for the evaluation, the label or comment is not used. The correctness is determined by the fact (in this example) that the subject and predicate appears in the KG. If a system produces exactly this fact, it is assumed that it also has the right semantics.

```JSON

{
    "text": "Gambas was designed by Benoît Minisini.",
    "triplets": [
        [
            "http://www.myiedata.org/uWZgJ000lFraOzsJn5Bc-bSrNO4=",
            "http://www.myiedata.org/CUhlOKxCqrwoJ82j33MyjrXU8rE=",
            "_:bXLsXnorV44G3wjesB2jWvQ7y7Bc="
        ]
    ],
    "additionalInfo": [
        [
            "_:bXLsXnorV44G3wjesB2jWvQ7y7Bc=",
            "http://www.w3.org/2000/01/rdf-schema#label",
            "Benoît Minisini"
        ],
        [
            "_:bXLsXnorV44G3wjesB2jWvQ7y7Bc=",
            "http://www.w3.org/2004/02/skos/core#altLabel",
            "Benoit Minisini"
        ],
        [
            "_:bXLsXnorV44G3wjesB2jWvQ7y7Bc=",
            "http://www.w3.org/2000/01/rdf-schema#comment",
            "French computer programmer"
        ],
        [
            "_:bXLsXnorV44G3wjesB2jWvQ7y7Bc=",
            "http://www.w3.org/1999/02/22-rdf-syntax-ns#type",
            "http://www.myiedata.org/WtXwuztT2NnnHaOb3DphZpkjEKQ="
        ]
    ]
}
```


For examples where no missing entity is involved, the field `additionalInfo` is empty:

```JSON

{
    "text": "Cry of Fear is a mod of the Half-Life video game, using the GoldSrc software engine.",
    "triplets": [
        [
            "http://www.myiedata.org/CFMU_rnbPmOAsKOGuzhwwMCHvOM=",
            "http://www.myiedata.org/X3HkdtI_0PBMyYZolXYgBA4xvcA=",
            "http://www.myiedata.org/1XDFvydjiTSkEDotL-Z2C1PJ5rg="
        ],
        [
            "http://www.myiedata.org/CFMU_rnbPmOAsKOGuzhwwMCHvOM=",
            "http://www.myiedata.org/MoA9DIZvR0DzTkTGvAdF-yOnook=",
            "http://www.myiedata.org/q9MnERbph9ei87PBwTQLiypJ6jk="
        ]
    ],
    "additionalInfo": []
}
```

