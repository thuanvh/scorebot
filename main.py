from typing import Union

from fastapi import FastAPI
from pydantic import BaseModel

class Item(BaseModel):
    message: str

app = FastAPI()


@app.get("/")
def read_root():
    return {"Hello": "World"}


@app.post("/chat")
def chat(item: Item):
    return {"message": item.message}