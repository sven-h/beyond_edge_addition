import pickle
from sentence_transformers import SentenceTransformer
import csv
import sys
import numpy as np

corpus_name = sys.argv[1]
print(f"start loading {corpus_name}.csv", flush=True)
corpus = []         
with open(f"{corpus_name}.csv", "r", newline="", encoding="utf-8") as file:
    csvFile = csv.reader(file)
    for i, line in enumerate(csvFile):
        corpus.append(line[1])
        #if i > 1000:
        #    break
print("loaded corpus", flush=True)
model = SentenceTransformer("all-MiniLM-L6-v2")
print("encode", flush=True)
corpus_embeddings = model.encode(corpus, convert_to_numpy=True)
print("encoding finished", flush=True)
#with open(f"{corpus_name}.pickle", "wb") as fOut:
#    pickle.dump(corpus_embeddings, fOut)
with open(f"{corpus_name}.npy", "wb") as fOut:
    np.save(fOut, corpus_embeddings)
print("file written", flush=True)