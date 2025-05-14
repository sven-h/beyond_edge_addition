import csv
import json
from typing import List, Union

import faiss
import numpy as np
from datasets import DatasetDict
from sentence_transformers import SentenceTransformer
from sklearn.preprocessing import normalize
from vllm import LLM
from vllm.lora.request import LoRARequest

from src.data_utils import create_full_entity_description
from src.special_prompts import RE_RET_PROMPT


def retrieve_candidates(texts: List[str], reps, index_mapping, model: Union[LLM, SentenceTransformer], k=5):

    embeddings = model.encode(texts, show_progress_bar=True, normalize_embeddings=True)

    scores = np.dot(embeddings, reps.T)
    nearest_indices = np.argsort(scores, axis=1)[:, -k:]
    all_candidates = []
    for nearest_indices_ in nearest_indices:
        candidates = []
        for idx in nearest_indices_:
            if idx in index_mapping:
                candidates.append(index_mapping[idx])
        all_candidates.append(candidates)
    return all_candidates


if __name__ == "__main__":

    input_file = "dev_v2.jsonl"
    prompt_to_add = RE_RET_PROMPT
    texts = []
    all_relation_reps = []
    index_mapping = {}
    encountered_pids = set()
    existing_pids = []
    for item in csv.reader(open("relation_schema.csv", "r")):
        pid, rel_label, rel_description = item
        rel_str = f"{rel_label}: {rel_description}"
        all_relation_reps.append(rel_str)
        index_mapping[len(all_relation_reps) - 1] = pid
        encountered_pids.add(pid)
    for item in open(input_file, "r"):
        data = json.loads(item)
        texts.append(prompt_to_add + data["text"])
        all_pids = set()
        for _, pid, _ in data["triples"]:
            if pid in encountered_pids:
                all_pids.add(pid)
        existing_pids.append(all_pids)
    model = SentenceTransformer("schema_retrieval_final/final", trust_remote_code=True)

    encoded_relations = model.encode(all_relation_reps, show_progress_bar=True, normalize_embeddings=True)

    candidates = retrieve_candidates(texts, encoded_relations, index_mapping, model)

    recalled = 0
    counter = 0
    for pid_list, candidates_found in zip(existing_pids, candidates):
        for pid in pid_list:
            if pid in {x for x in candidates_found}:
                recalled += 1
            counter += 1
    print(f"Recall: {recalled / counter}")