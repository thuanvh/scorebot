from qdrant_client import QdrantClient

from langchain_community.embeddings import LlamaCppEmbeddings
import pandas as pd
import numpy as np
from qdrant_client.models import PointStruct
from qdrant_client.models import Distance, VectorParams
import os

class DbNpy:
    def __init__(self,llm) -> None:
        # Data for reference
        self.df = pd.read_csv("data.csv")
        print(df)

        with open('output.npy','rb') as f:
            self.doc_data = np.load(f)

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
        # query = "Xin chào"
        # query = "Tỉ số trận Manchester và Nottingham"
        query = query_text
        query_embed = self.llm.embed_query(query)
        print(query_embed)

        #scores = (doc_db[:] @ query_embed[:].T) * 100
        #print(scores.tolist())

        #a=np.array(doc_db)
        b = np.array(query_embed)

        scores =  self.doc_data[:] @ b[:].T
        #print(scores)
        max_index = np.argmax(scores)
        #print(max_index, scores[max_index])

        #print(df['text'][max_index],df['action'][max_index])
        action =  self.df['action'][max_index]
        print(action)
