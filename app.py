import gradio as gr
import os
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, AutoModel

# download qiaofeng_chatbot to the base_path directory using git tool
base_path = './qiaofeng_chatbot'
os.system(f'git clone https://code.openxlab.org.cn/Godfrey/qiaofeng_chatbot.git {base_path}')
os.system(f'cd {base_path} && git lfs pull')

tokenizer = AutoTokenizer.from_pretrained(base_path,trust_remote_code=True)
model = AutoModelForCausalLM.from_pretrained(base_path,trust_remote_code=True, torch_dtype=torch.float16).cuda()

def chat(message,history):
    for response,history in model.stream_chat(tokenizer,message,history,max_length=2048,top_p=0.7,temperature=1):
        yield response

gr.ChatInterface(chat,
                 title="qiaofeng_chatbot",
                description="""
qiaofeng_chatbot is developed by Godfrey based on internlm2-chat-1_8b.  
                 """,
                 ).queue(1).launch()