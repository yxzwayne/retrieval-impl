import replicate
import json
import dotenv

dotenv.load_dotenv(dotenv_path=".env.local")


with open("results/results.json") as f:
    results = json.load(f)

query = "protoss has the ability to build two buildings at the same time"

input_list = [[query, result["chunk"]] for result in results]


input = {"input_list": json.dumps(input_list)}

output = replicate.run(
    "yxzwayne/bge-reranker-v2-m3:7f7c6e9d18336e2cbf07d88e9362d881d2fe4d6a9854ec1260f115cabc106a8c",
    input=input,
)
# print(output)
# => [-11.03125,-2.306640625,-10.40625]

import numpy as np

scores = np.array(output)
rank = np.argsort(scores)[::-1]
top_10_results = [results[i] for i in rank[:10]]

# save the top 10 into new file
with open("top_10_reranked.json", "w") as f:
    json.dump(top_10_results, f)
