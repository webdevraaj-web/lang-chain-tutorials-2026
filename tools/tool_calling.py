from langchain_community.tools import tool
from langchain.messages import HumanMessage
from langchain_groq import ChatGroq
from dotenv import load_dotenv
import os

load_dotenv('.env')
# create tool
@tool
def multiply(a:int,b:int)->int:
    ''' this tools is related to multification a ot b'''
    return a*b

# tool binding

GROQ_KEY=os.getenv('GROQ_API_KEY')
llm=ChatGroq(model="llama-3.3-70b-versatile" ,api_key=GROQ_KEY)
llm_with_tools=llm.bind_tools([multiply])

query=HumanMessage('can you multiply the 3 and 30?')

message=[query]

llm_result=llm_with_tools.invoke(message)

message.append(llm_result)

# tool execution

tool_execution=multiply.invoke(llm_result.tool_calls[0])


message.append(tool_execution)

llm_message=llm_with_tools.invoke(message)


print(llm_message.content)
 


 