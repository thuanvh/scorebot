from qdrant_client import QdrantClient

from langchain_community.embeddings import LlamaCppEmbeddings
import pandas as pd
import numpy as np
from qdrant_client.models import PointStruct
from qdrant_client.models import Distance, VectorParams
import os

class DbQdrant:
    def __init__(self,llm) -> None:
        

        modelmap = {"vistral":{
            "file":"ggml-vistral-7B-chat-q4_0.gguf",
            "db":"qdrant_vistral",
            "size":4096
        },
        }
        model_name="vistral"
        self.model = modelmap[model_name]
        #llama = LlamaCppEmbeddings(model_path=model["file"])

        self.collection_name="score_collection"


        # Initialize the client
        #client = QdrantClient(":memory:") 
        self.client = QdrantClient(path=self.model["db"])
        self.llm = llm

    def generate_db(self):
        if not os.path.exists(self.model["file"]):
            
            df = pd.read_csv("data.csv")
            print(df)
            doc_result = self.llm.embed_documents(df["text"])
            print(doc_result)
            print(len(doc_result),len(doc_result[0]))

            self.client.recreate_collection(
                collection_name=self.collection_name,
                vectors_config=VectorParams(size=self.model['size'], distance=Distance.COSINE),
            )
            # Use the new add method
            self.client.upsert(
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

    def search(self,query_text, count=1):
        query_vector = self.llm.embed_query(query_text)
        search_result = self.client.search(
            collection_name=self.collection_name,
            query_vector=query_vector,
            limit=count,
            with_payload=True
        )
        print(search_result)
        return search_result