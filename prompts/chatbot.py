from langchain_huggingface import ChatHuggingFace,HuggingFaceEndpoint
from dotenv import load_dotenv
from langchain_core.messages import SystemMessage,HumanMessage,AIMessage
load_dotenv()

llm=HuggingFaceEndpoint(
    repo_id="MiniMaxAI/MiniMax-M2.7",
    task="text-generation",
)
model=ChatHuggingFace(llm=llm)


chat_history=[
    SystemMessage(content="hii im bot")
]
while True:
    user_input=input('User: ')
    chat_history.append(HumanMessage(content=user_input))
    if user_input == 'exit':
        break

    result=model.invoke(chat_history)
    chat_history.append(AIMessage(content=result.content))
    print('ai: ', result.content)


print(chat_history)

     
