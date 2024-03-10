import llama_cpp
import llama_cpp.llama_tokenizer

if __name__ == "__main__":

    llama = llama_cpp.Llama.from_pretrained(
        repo_id="sail/Sailor-0.5B-Chat-gguf",
        filename="ggml-model-Q4_K_M.gguf",
        tokenizer=llama_cpp.llama_tokenizer.LlamaHFTokenizer.from_pretrained("sail/Sailor-0.5B-Chat"),
        n_gpu_layers=0,
        n_threads=4,
        verbose=False,
    )

    system_role= 'system'
    user_role = 'question'
    assistant_role = "answer"

    sft_start_token =  "<|im_start|>"
    sft_end_token = "<|im_end|>"
    ct_end_token = "<|endoftext|>"

    system_prompt= \
    'Hãy tìm tên hai đội bóng đá từ câu hỏi và trả lời dạng json ```json {""team1"":  Tên đội bóng đá 1, ""team2"": Tên đội bóng đá 2}. Hãy để trống tên đội bóng nếu không có thông tin.'
    system_prompt = f"<|im_start|>{system_role}\n{system_prompt}<|im_end|>"
    print(system_prompt)
    

    message ='Hãy tìm tên hai đội bóng trong câu sau: "Tỉ số trận Manchester United và Liverpool hôm qua" '
    history =[]
    history_transformer_format = history + [[message, ""]]
    print(history_transformer_format)
    # Formatting the input for the model.
    messages = system_prompt + sft_end_token.join([
        sft_end_token.join([
            f"\n{sft_start_token}{user_role}\n" + item[0],
            f"\n{sft_start_token}{assistant_role}\n" + item[1]
        ]) for item in history_transformer_format
    ])
    print(messages)
    response = llama(
        messages,
        stream=False,
        top_p=0.75,
        top_k=60,
        stop=["<|im_end|>", "<|endoftext|>"],
        temperature=0.2,
        max_tokens=256,
    )
    print(response)
    text = ""
    
    res = response["choices"][0]
    
    text += res["text"]
    print(text)
    # modelfile='''
    # FROM llama2
    # SYSTEM You are mario from super mario bros.
    # '''

    # ollama.create(model='example', modelfile=modelfile)

    # modelfile='''
    # FROM llama2
    # '''
    # ollama.create(model='example', modelfile=modelfile)

    # response = ollama.chat(model='sample', messages=[
    #   {
    #     'role': 'user',
    #     'content': 'Why is the sky blue?',
    #   },
    # ])
    # print(response['message']['content'])
# class OllamaClient:
#     def chat(self, message:str,system:str):
#         modelname='gemma:2b'
#         response = ollama.chat(model=modelname, messages=[
#             {'role': 'system','content': system,},
#             {'role': 'user','content': message,},
            
#         ])
#         print(response['message'])
#         return response['message']['content']