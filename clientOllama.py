import ollama

if __name__ == "__main__":
    
    modelname='gemma:2b'
    response = ollama.chat(model=modelname, messages=[
        {'role': 'system','content': 'Hãy tổng hợp thông tin về tên hai đội bóng đá từ người dùng và chỉ trả lời dạng json ```json {""team1"":  Tên đội bóng đá 1, ""team2"": Tên đội bóng đá 2}. Hãy để trống tên đội bóng nếu không có thông tin',},
        {'role': 'user','content': 'Tỉ số trận Manchester United và Liverpool hôm qua',},
        
    ])
    print(response['message'])

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
class OllamaClient:
    def chat(self, message:str,system:str):
        modelname='gemma:2b'
        response = ollama.chat(model=modelname, messages=[
            {'role': 'system','content': system,},
            {'role': 'user','content': message,},
            
        ])
        print(response['message'])
        return response['message']['content']