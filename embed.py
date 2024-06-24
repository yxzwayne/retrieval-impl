import jsonlines
import numpy as np
from tqdm import tqdm
from mlx_embedding_models.embedding import EmbeddingModel

emb_model_name = "gte-tiny"

model = EmbeddingModel.from_registry(emb_model_name)

chunks = []
with jsonlines.open("data/iodis_chunks.jsonl") as reader:
    for obj in reader:
        chunks.append(obj["chunk"])

n_batch = 128
batched_chunks = [chunks[i : i + n_batch] for i in range(0, len(chunks), n_batch)]

embs = []
for batch in tqdm(batched_chunks):
    embs.extend(model.encode(batch))

# save embedding to disk
embs = np.array(embs)
np.save("data/chunk_embs.npy", embs)
