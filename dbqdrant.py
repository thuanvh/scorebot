from qdrant_client import QdrantClient

from langchain_community.embeddings import LlamaCppEmbeddings
import pandas as pd
import numpy as np
from qdrant_client.models import PointStruct
from qdrant_client.models import Distance, VectorParams

df = pd.read_csv("data.csv")
print(df)

modelmap = {"vistral":{
    "file":"ggml-vistral-7B-chat-q4_0.gguf",
    "db":"qdrant_vistral",
    "size":4096
},
}
model_name="vistral"
model = modelmap[model_name]
llama = LlamaCppEmbeddings(model_path=model["file"])

doc_result = llama.embed_documents(df["text"])
print(doc_result)
print(len(doc_result),len(doc_result[0]))


# Initialize the client
#client = QdrantClient(":memory:") 
client = QdrantClient(path=model["db"])

# Prepare your documents, metadata, and IDs
# docs = ["Qdrant has Langchain integrations", "Qdrant also has Llama Index integrations"]
# metadata = [
#     {"source": "Langchain-docs"},
#     {"source": "Linkedin-docs"},
# ]
# ids = [42, 2]
collection_name="score_collection"
client.recreate_collection(
    collection_name=collection_name,
    vectors_config=VectorParams(size=model['size'], distance=Distance.COSINE),
)
# Use the new add method
client.upsert(
    collection_name="score_collection",
    points=[
        PointStruct(
            id=int(df['id'][idx]),
            vector=vector,
            payload={"text": df["text"][idx], "label": df["label"][idx]}
        )
        for idx, vector in enumerate(doc_result)
    ]
)



query_text="This is a query document"
query_vector = llama.embed_query(query_text)
search_result = client.search(
    collection_name=collection_name,
    query_vector=query_vector,
    limit=5  # Return 5 closest points
)
print(search_result)