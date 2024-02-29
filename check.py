from langchain_community.embeddings import LlamaCppEmbeddings
import pandas as pd
import numpy as np


# Data for reference
df = pd.read_csv("data.csv")
print(df)

llama = LlamaCppEmbeddings(model_path="ggml-vistral-7B-chat-q4_0.gguf")

with open('output.npy','rb') as f:
    doc_data = np.load(f)


query = "Xin chào"
query = "Tỉ số trận Manchester và Nottingham"
query_embed = llama.embed_query(query)
print(query_embed)

#scores = (doc_db[:] @ query_embed[:].T) * 100
#print(scores.tolist())

#a=np.array(doc_db)
b = np.array(query_embed)

scores = doc_data[:] @ b[:].T
print(scores)
max_index = np.argmax(scores)
print(max_index, scores[max_index])

print(df['text'][max_index],df['action'][max_index])