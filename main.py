from typing import Union

from fastapi import FastAPI
from pydantic import BaseModel

from langchain_community.embeddings import LlamaCppEmbeddings
import pandas as pd
import numpy as np

from dbqdrant import DbQdrant


class Item(BaseModel):
    message: str

app = FastAPI()

llama = LlamaCppEmbeddings(model_path="ggml-vistral-7B-chat-q4_0.gguf")


db = DbQdrant(llama)


@app.get("/")
def read_root():
    return {"Hello": "World"}


@app.post("/chat")
def chat(item: Item):
    res = db.search(item.message)
    print(res[0])
    print(res[0].id)
    print(res[0].score)
    print(res[0].payload)
    action = res[0].payload['label']
    if action == 'score_specific':
        reply = score_specific(item.message)
    elif action == 'hello':
        reply = hello(item.message)
    elif action == 'help_bet':
        reply = help_bet(item.message)
    elif action == 'help_score':
        reply = help_score(item.message)
    return {"message": item.message, "similar_text":res[0].payload['text'],
            "similarity":res[0].score,
            "label": action, "reply":reply}

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
    ###
    #     curl http://localhost:11434/api/chat -d '{
    #   "model": "gemma:2b",
    #   "messages": [
    #     { "role": "system", "content": "Hãy tổng hợp thông tin về tên hai đội và chỉ trả lời dạng json ```json {\"team1\":  Tên đội 1, \"team2\": Tên đội 2}. Hãy để trống nếu không có thông tin" },
    #     { "role": "user", "content": "Kết quả trận đấu giữa Manchester City và Liverpool?" }
    #   ]
    # }'

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
