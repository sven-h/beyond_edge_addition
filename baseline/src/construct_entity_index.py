import dataclasses
import json
import pickle
from argparse import ArgumentParser

import faiss
import numpy
from sentence_transformers import SentenceTransformer
from transformers import AutoTokenizer, AutoModelForCausalLM
from vllm import LLM
from vllm.lora.request import LoRARequest

from src.data_utils import KGContainer, create_full_entity_description
from sklearn.preprocessing import normalize

def create_hnsw_index(encoded):
    d = encoded.shape[1]
    M = 64  # Number of connections per element in the graph (controls the accuracy)
    efConstruction = 200  # Controls how the graph is built (higher = more accurate but slower build)
    efSearch = 128  # Controls the search time/accuracy trade-off during querying

    # Create the HNSW index
    index = faiss.IndexHNSWFlat(d, M)  # M: number of connections per vector
    index.hnsw.efConstruction = efConstruction  # Set the efConstruction value
    # index = faiss.IndexFlatIP(d)  # Use inner product for similarity search

    # Add the data to the index
    index.add(encoded)  # Add the entire dataset to the index

    # Set search parameters
    index.hnsw.efSearch = efSearch  # Set the efSearch value
    return index

def create_ivf_hnsw_index(encoded):
    d = encoded.shape[1]
    N = encoded.shape[0]

    # Dynamically adjust based on number of vectors
    if N < 100_000:
        nlist = 256
        M = 16
        nprobe = 16
    elif N < 1_000_000:
        nlist = 1024
        M = 32
        nprobe = 32
    elif N < 10_000_000:
        nlist = 4096
        M = 48
        nprobe = 64
    else:
        nlist = 8192
        M = 64
        nprobe = 128

    # Create HNSW-based quantizer for clustering
    quantizer = faiss.IndexHNSWFlat(d, M)
    quantizer.hnsw.efConstruction = max(40, M * 2)

    # Create IVF index with inner product metric
    index = faiss.IndexIVFFlat(quantizer, d, nlist, faiss.METRIC_INNER_PRODUCT)

    # Train the index (KMeans on quantizer)
    index.train(encoded)

    # Add the dataset to the index
    index.add(encoded)

    # Set nprobe for query-time accuracy
    index.nprobe = nprobe

    return index


def create_flat_index(encoded):
    d = encoded.shape[1]
    quantizer = faiss.IndexFlatIP(d)
    nlist = 100  # Number of cells / clusters
    index_ivf = faiss.IndexIVFFlat(quantizer, d, nlist, faiss.METRIC_INNER_PRODUCT)

    index_ivf.train(encoded)  # You must train first
    index_ivf.add(encoded)

    # Then query:
    index_ivf.nprobe = 10  # How many cells to probe at query time
    return index_ivf
def construct(model, kgc, model_type, adapter_path, index_name):
    identifiers = []
    text_reps = []
    index_to_entity = {}
    for entity_key, entity in kgc.entities.items():
        if entity_key not in kgc.relations:
            identifiers.append(entity_key)
            text_reps.append(create_full_entity_description(entity.label, entity.definition, [kgc.label(x) for x in entity.types if x in kgc.entities]))
            entity_as_dict = dataclasses.asdict(entity)
            entity_as_dict["types"] = [kgc.label(x) for x in entity.types if x in kgc.entities]
            index_to_entity[len(identifiers) - 1] = entity_as_dict

    if model_type == "sts":
        encoded = model.encode(text_reps, show_progress_bar=True, normalize_embeddings=True)
    else:
        if adapter_path:
            encoded = model.embed(text_reps, lora_request=LoRARequest(adapter_path, hash(adapter_path), adapter_path))
        else:
            encoded = model.embed(text_reps)
        encoded = numpy.array([x.outputs.embedding for x in encoded], dtype=numpy.float32)

        # Normalize the embeddings
        encoded = normalize(encoded, axis=1)

    pickle.dump(encoded, open(f"{index_name}.pkl", "wb"))
    json.dump(index_to_entity, open(f"{index_name}.json", "w"), indent=4)

    d = encoded.shape[1]
    index = create_ivf_hnsw_index(encoded)  # Use the flat index

    # Save the index to disk
    faiss.write_index(index, f"{index_name}.index")






def load_model(model_name, model_type):
    assert model_type in ["sts", "hf", "vllm"]  # Either a sentence transformer or a huggingface LLM
    if model_type == "hf":
        model, tokenizer = (
            AutoModelForCausalLM.from_pretrained(model_name, device_map="auto"),
            AutoTokenizer.from_pretrained(model_name),
        )
    elif model_type == "sts":
        model = SentenceTransformer(model_name, trust_remote_code=True)
        tokenizer = None
    elif model_type == "vllm":
        tokenizer = None
        model = LLM(model=model_name, task="embed", enable_lora=True)

    else:
        raise ValueError(f"Unknown model type {model_type}")
    return model, tokenizer


if __name__ == "__main__":
    argparser = ArgumentParser()
    argparser.add_argument(
        "--el_embedder_name",
        default="intfloat/e5-mistral-7b-instruct",
        help="Embedding model used for entity linking. Has to be a sentence transformer. Please refer to https://sbert.net/",
    )
    argparser.add_argument(
        "--el_adapter_path",
        default=None,
        help="Path to adapter of entity linker.",
    )
    argparser.add_argument("--model_type", default="vllm", help="Model type: vllm, hf, or sts")
    argparser.add_argument("--index_name", default="entity_index", help="Name of the index file to save")
    kgc = KGContainer("data/ie_dataset_v3/")

    args = argparser.parse_args()
    model_type = args.model_type
    el_embedder, tokenizer = load_model(args.el_embedder_name, model_type)


    construct(el_embedder, kgc, model_type, args.el_adapter_path, args.index_name)
