# vector similarity search via faiss
import faiss
import numpy as np
import jsonlines
from openai import OpenAI
import dotenv
import os

dotenv.load_dotenv(dotenv_path=".env.local")

client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

embs = np.load("data/chunk_embs.npy")

print(f"dimensions of embeddings: {embs.shape}")
emb_dim = embs.shape[1]

video_ids = []
chunks = []
print("- Importing data from json file")
with jsonlines.open("data/iodis_chunks.jsonl") as reader:
    for obj in reader:
        video_ids.append(obj["video_id"])
        chunks.append(obj["chunk"])


print("- Building FAISS index")
index = faiss.IndexFlatL2(emb_dim)
index.add(embs)


print("- Embedding query")
query = "protoss has the ability to build two structures with one probe but terran needs two scvs and zerg needs two drones"

q_emb = np.array(
    client.embeddings.create(
        model="text-embedding-3-small", input=query, dimensions=emb_dim
    )
    .data[0]
    .embedding
)

print("- Searching for similar videos")
distances, indices = index.search(np.array([q_emb]), 100)

result = [{"video_id": video_ids[i], "chunk": chunks[i]} for i in indices[0]]

import json

with open(f"results/results.json", "w") as f:
    json.dump(result, f)
