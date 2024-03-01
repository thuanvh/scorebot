from typing import Union

from fastapi import FastAPI
from pydantic import BaseModel

from langchain_community.embeddings import LlamaCppEmbeddings
import pandas as pd
import numpy as np


class Item(BaseModel):
    message: str

app = FastAPI()



# Data for reference
df = pd.read_csv("data.csv")
print(df)

llama = LlamaCppEmbeddings(model_path="ggml-vistral-7B-chat-q4_0.gguf")

with open('output.npy','rb') as f:
    doc_data = np.load(f)




@app.get("/")
def read_root():
    return {"Hello": "World"}


@app.post("/chat")
def chat(item: Item):
    # query = "Xin chào"
    # query = "Tỉ số trận Manchester và Nottingham"
    query = item.message
    query_embed = llama.embed_query(query)
    print(query_embed)

    #scores = (doc_db[:] @ query_embed[:].T) * 100
    #print(scores.tolist())

    #a=np.array(doc_db)
    b = np.array(query_embed)

    scores = doc_data[:] @ b[:].T
    #print(scores)
    max_index = np.argmax(scores)
    #print(max_index, scores[max_index])

    #print(df['text'][max_index],df['action'][max_index])
    action = df['action'][max_index]
    print(action)
    if action == 'score_specific':
        reply = score_specific(item.message)
    elif action == 'hello':
        reply = hello(item.message)
    elif action == 'help_bet':
        reply = help_bet(item.message)
    elif action == 'help_score':
        reply = help_score(item.message)
    return {"message": item.message, "similar_text":df['text'][max_index],"similarity":scores[max_index], "action": df['action'][max_index], "reply":reply}

def call_lmstudio(message:str, system:str):
    # Example: reuse your existing OpenAI setup
    from openai import OpenAI

    # Point to the local server
    client = OpenAI(base_url="http://192.168.111.1:1234/v1", api_key="not-needed")

    completion = client.chat.completions.create(
    model="local-model", # this field is currently unused
    messages=[
        {"role": "system", "content": system},
        {"role": "user", "content": message}
    ],
    temperature=0.7,
    )

    print(completion.choices[0].message)
    return completion.choices[0].message

def help_score(message : str):
    content = "Bạn cung cấp thông tin tỉ số trận đấu. Người dùng cần cung cấp tên hai đội bóng."
    return call_lmstudio(message, content)

def help_bet(message : str):
    content = "Bạn cung cấp thông tin đặt cược trận đấu. Người dùng cần cung cấp tên hai đội bóng."
    return call_lmstudio(message, content)

def score_specific(message : str):
    content = "Hãy tổng hợp thông tin về tên hai đội bóng đá từ người dùng và chỉ trả lời dạng json ```json {""team1"":  Tên đội bóng đá 1, ""team2"": Tên đội bóng đá 2}. Hãy để trống tên đội bóng nếu không có thông tin"
    return call_lmstudio(message, content)
    # # Example: reuse your existing OpenAI setup
    # from openai import OpenAI

    # # Point to the local server
    # client = OpenAI(base_url="http://192.168.111.1:1234/v1", api_key="not-needed")

    # completion = client.chat.completions.create(
    # model="local-model", # this field is currently unused
    # messages=[
    #     {"role": "system", "content": "Hãy tổng hợp thông tin về tên hai đội bóng đá từ người dùng và chỉ trả lời dạng json ```json {""team1"":  Tên đội bóng đá 1, ""team2"": Tên đội bóng đá 2}. Hãy để trống tên đội bóng nếu không có thông tin"},
    #     {"role": "user", "content": message}
    # ],
    # temperature=0.7,
    # )

    # print(completion.choices[0].message)
    # return completion.choices[0].message

def hello(message: str):
    content = "Bạn là trợ lý ảo hữu dụng. Bạn cung cấp các thông tin tỉ số các trận đấu. Hãy chào hỏi mọi người và nói về chức năng của bạn."
    return call_lmstudio(message, content)
    # from openai import OpenAI

    # # Point to the local server
    # client = OpenAI(base_url="http://192.168.111.1:1234/v1", api_key="not-needed")

    # completion = client.chat.completions.create(
    # model="local-model", # this field is currently unused
    # messages=[
    #     {"role": "system", "content": "Bạn là trợ lý ảo hữu dụng. Bạn cung cấp các thông tin tỉ số các trận đấu. Hãy chào hỏi mọi người và nói về chức năng của bạn."},
    #     {"role": "user", "content": message}
    # ],
    # temperature=0.7,
    # )

    # print(completion.choices[0].message)
    # return completion.choices[0].message
