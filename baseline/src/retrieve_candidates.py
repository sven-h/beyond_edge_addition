import json
import pickle
from argparse import ArgumentParser
from typing import List, Union

import faiss
import numpy as np
from sentence_transformers import SentenceTransformer
from sklearn.preprocessing import normalize
from tqdm import tqdm
from vllm import LLM
from vllm.lora.request import LoRARequest

from src.data_utils import create_full_entity_description
from src.special_prompts import STS_PROMPT


def retrieve_candidates(texts: List[str], faiss_index, index_mapping, model: Union[LLM, SentenceTransformer], adapter_path: str, k=10):
    if isinstance(model, LLM):
        if adapter_path:
            embeddings = model.embed(texts, lora_request=LoRARequest(adapter_path, hash(adapter_path), adapter_path))
        else:
            embeddings = model.embed(texts)
        embeddings = [x.outputs.embedding for x in embeddings]
        embeddings = np.array(embeddings, dtype=np.float32)
        embeddings = normalize(embeddings, axis=1)
    else:
        embeddings = model.encode(texts, show_progress_bar=True, normalize_embeddings=True)

    if isinstance(faiss_index, np.ndarray):
        numpy_array = faiss_index
        batch_size = 1000
        nearest_indices = []
        for i in tqdm(range(0, len(embeddings), batch_size)):
            batch_embeddings = embeddings[i:i + batch_size]
            scores = batch_embeddings @ numpy_array.T
            indices = np.argsort(scores, axis=1)[:, -k:]
            nearest_indices.extend(indices)
        nearest_indices = np.array(nearest_indices)
    else:
        nearest_indices = faiss_index.search(embeddings, k)[1]
    all_candidates = []
    for nearest_indices_ in nearest_indices:
        candidates = []
        for idx in nearest_indices_:
            if idx in index_mapping:
                candidates.append(index_mapping[idx])
        all_candidates.append(candidates)
    return all_candidates


if __name__ == "__main__":
    argparser = ArgumentParser()
    argparser.add_argument("input_file", type=str)
    argparser.add_argument("index_name", type=str)
    argparser.add_argument("model_name", type=str)
    args = argparser.parse_args()
    input_file = args.input_file
    adapter_path = "e5-mistral-7b-instruct/candidate_trained_model"
    prompt_to_add = STS_PROMPT
    texts = []
    target_entity_list = []
    for item in open(input_file, "r"):
        data = json.loads(item)
        for entity in data["generated_entities"]:
            texts.append(prompt_to_add + create_full_entity_description(entity["surface_form"], entity["definition"], [entity["type"]]))
            target_entity_list.append(entity["entity"])
    faiss_index = faiss.read_index(f"{args.index_name}.index")  # Replace with actual index
    #faiss_index = pickle.load(open("entity_index_v3.pkl", "rb"))  # Replace with actual index
    index_mapping = json.load(open(f"{args.index_name}.json"))  # Replace with actual index mapping
    index_mapping = {int(k): v for k, v in index_mapping.items()}
    # model = LLM(model="intfloat/e5-mistral-7b-instruct", task="embed", enable_lora=True, gpu_memory_utilization=0.45)
    model = SentenceTransformer(args.model_name, trust_remote_code=True)

    candidates = retrieve_candidates(texts, faiss_index, index_mapping, model, adapter_path)

    recalled = 0
    in_index = 0
    entity_set = {x["label"] for x in index_mapping.values()}
    for entity, candidates_found in zip(target_entity_list, candidates):
        if entity in {x["label"] for x in candidates_found}:
            recalled += 1
        if entity in entity_set:
            in_index += 1
    print(f"Recall: {recalled / len(target_entity_list)}")
    print(f"Recall: {in_index / len(target_entity_list)}")