import numpy as np
import pickle
import faiss
from datetime import datetime
from collections import defaultdict
from tqdm import tqdm
import csv
import sys
from rdflib import URIRef

corpus_name = sys.argv[1]

print(f"{datetime.now()} load corpus", flush=True)
uri_to_ids = defaultdict(set)
id_to_uri = []
with open(f"{corpus_name}.csv", "r", newline="", encoding="utf-8") as f:
    reader = csv.reader(f)
    for i, row in enumerate(reader):
        entity_uri = row[0]
        uri_to_ids[entity_uri].add(i)
        id_to_uri.append(entity_uri)
        if i % 1000000 == 0:
            print(f"{datetime.now()} load corpus - {i}", flush=True)


print(f"{datetime.now()} load embeddings", flush=True)
corpus_embeddings = np.load(f"{corpus_name}.npy")
print(f"{datetime.now()} normalize embeddings", flush=True)
corpus_embeddings = corpus_embeddings / np.linalg.norm(corpus_embeddings, axis=1)[:, None]

print(f"{datetime.now()} build index", flush=True)

d = 384                       # Dimensionality

# --- IVF Parameters ---
nlist = 65536                 # Number of Voronoi cells (coarse centroids)

# --- PQ Parameters ---
m = 48                        # Number of subquantizers:   d (384) must be divisible by m. 384 / 48 = 8 (sub-vector size).
nbits = 8                     # Bits per sub-quantizer code. Almost always 8 for IndexIVFPQ.

quantizer = faiss.IndexFlatIP(d)
index = faiss.IndexIVFPQ(quantizer, d, nlist, m, nbits, faiss.METRIC_INNER_PRODUCT)

res = faiss.StandardGpuResources()
index = faiss.index_cpu_to_gpu(res, 0, index)

print(f"{datetime.now()} train with 20 million", flush=True)

train_sample = corpus_embeddings[np.random.choice(len(corpus_embeddings), 20_000_000, replace=False)]
index.train(train_sample)

print(f"{datetime.now()} add vectors", flush=True)
batch_size = 1_000_000
for i in range(0, len(corpus_embeddings), batch_size):
    end = i + batch_size
    index.add(corpus_embeddings[i:end])
    
index.nprobe = 512 # 512 is the number of Voronoi cells to search in the coarse quantizer - initially 1024 but too slow

print(f"{datetime.now()} index finished", flush=True)



with open("kg_info.pickle", 'rb') as f:
    object_to_load = pickle.load(f)
    classes_that_are_instances = object_to_load["classes_that_are_instances"]
    all_instances = object_to_load["all_instances"]
    
# extract which entities to search for
with open("dataset_info.pickle", 'rb') as f:
    object_to_load = pickle.load(f)
    train_dataset = object_to_load["train_dataset"]
    new_val_dataset = object_to_load["new_val_dataset"]
    new_test_dataset = object_to_load["new_test_dataset"]
    removed_instances_in_train = object_to_load["removed_instances_in_train"]
    removed_instances_in_val = object_to_load["removed_instances_in_val"]
    removed_instances_in_test = object_to_load["removed_instances_in_test"]
    
removed_instances = removed_instances_in_train | removed_instances_in_val | removed_instances_in_test

entities_appearing_in_dataset = set()
for dataset in [train_dataset, new_val_dataset, new_test_dataset]:
    for text, triples in dataset:
        for s, p, o in triples:
            if s in all_instances or s in classes_that_are_instances:
                entities_appearing_in_dataset.add(s)
            if o in all_instances or o in classes_that_are_instances:
                entities_appearing_in_dataset.add(o)


# here we still include the removed ones, because we also want to find for removed entities very similar other entities

print(f"{datetime.now()} sample negatives", flush=True)

hard_negatives = set()
for search_candidate in tqdm(entities_appearing_in_dataset):
    search_ids = uri_to_ids[str(search_candidate)]
    if len(search_ids) == 0:
        print(f"{datetime.now()} don't find embeddings for {search_candidate}", flush=True)
    
    query_embeddings = corpus_embeddings[list(search_ids)]

    found_distances, found_ids = index.search(query_embeddings, 10)
    # sort all hits by distance descending
    sorted_hits = []
    for i in range(len(found_ids)):
        for j in range(len(found_ids[i])):
            sorted_hits.append((found_distances[i][j], found_ids[i][j]))
    sorted_hits = sorted(sorted_hits, key=lambda x: x[0], reverse=True)
    
    
    for hit in sorted_hits[:10]:
        hard_negatives.add(URIRef(id_to_uri[hit[1]]))

    #for x in sorted_hits[:10]:
    #    print(f"Distance: {x[0]}")
    #    print(f"Index: {x[1]}")
    #    print(f"Label: {corpus_full[x[1]]}")
    #    print()
  
hard_negatives = hard_negatives - removed_instances
#save to file
with open(f"{corpus_name}_hard_negatives.pickle", "wb") as f:
    pickle.dump(hard_negatives, f)