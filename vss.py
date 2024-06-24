# vector similarity search via faiss

import numpy as np
import faiss

embs = np.load("data/chunk_embs.npy")

index = faiss.IndexFlatL2(embs.shape[1])
index.add(embs)

print(index.search(embs[:1], 1))