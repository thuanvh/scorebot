from langchain_community.embeddings import LlamaCppEmbeddings

llama = LlamaCppEmbeddings(model_path="ggml-vistral-7B-chat-q4_0.gguf")

doc_result = llama.embed_documents([text])
print(doc_result)

