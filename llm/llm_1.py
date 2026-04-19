from langchain_huggingface import ChatHuggingFace,HuggingFaceEndpoint
from dotenv import load_dotenv
import os

load_dotenv('.env')

llm=HuggingFaceEndpoint(
    repo_id="MiniMaxAI/MiniMax-M2.7",
    task="text-generation"
)
model=ChatHuggingFace(llm=llm) 

result=model.invoke('what is the capital of the india')

print(result.content)