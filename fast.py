from langchain_community.embeddings import LlamaCppEmbeddings
import pandas as pd
import numpy as np

df = pd.read_csv("data.csv")
print(df)

llama = LlamaCppEmbeddings(model_path="ggml-vistral-7B-chat-q4_0.gguf")

doc_result = llama.embed_documents(df["text"])
print(doc_result)

with open('output.npy','wb') as f:
    np.save(f, doc_result)

