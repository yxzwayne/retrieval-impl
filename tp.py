import os
import time
import jsonlines
from tqdm import tqdm
from dotenv import load_dotenv
from trufflepig import Trufflepig

load_dotenv(dotenv_path=".env.local")

api_key = os.getenv("TP_API_KEY")


if __name__ == "__main__":
    client = Trufflepig(api_key)
    existing_idx_names = [idx.index_name for idx in client.list_indexes()]
    if "iodis" not in existing_idx_names:
        index = client.create_index("iodis")
        print(f"Created index {index.index_name}")
    else:
        index = client.get_index("iodis")
        print(f"Delete previous index {index.index_name}")
        client.delete_index(index.index_name)
        index = client.create_index("iodis")
        print(f"Created index {index.index_name}")
    data = []

    with jsonlines.open("data/iodis_chunks.jsonl") as reader:
        for obj in reader:
            data.append(obj)

    n_batch = 5

    for i in tqdm(range(0, len(data), n_batch)):
        batch = data[i : i + n_batch]
        upload_response = index.upload(
            text=[
                {
                    "document_key": obj["video_id"],
                    "document": obj["chunk"],
                }
                for obj in batch
            ]
        )
        print(f"Queued batch {i//n_batch + 1} for upload")
        time.sleep(1)

    search_result = index.search(
        "protoss can build two buildings with just one probe protoss is imba just kidding"
    )
    for i, r in enumerate(search_result):
        print(f"Rank {i}")
        print(f"   Document {r.document_key}")
        print(f"   Content: {r.content}")
