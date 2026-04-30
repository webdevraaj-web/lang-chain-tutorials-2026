from langchain_community.tools import tool
from langchain.messages import HumanMessage
from langchain_groq import ChatGroq
from dotenv import load_dotenv
import os
import requests
from langchain.tools import InjectedToolArg
from typing import Annotated
import json
load_dotenv('.env')
 

GROQ_KEY=os.getenv('GROQ_API_KEY')
EXCHANGE_RATE_API=os.getenv('EXCHANGE_RATE_API')

 

 

# create tool

@tool
def get_conversion_factor(base_currency:str,target_currency:str)->float:
    '''
    this tool is related to given base currency to target currenct conversion related
    '''
    url=f'https://v6.exchangerate-api.com/v6/{EXCHANGE_RATE_API}/pair/{base_currency}/{target_currency}'
    response=requests.get(url)
    return response.json()
    

@tool
def convert(base_currency_value:int,conversion_rate:Annotated[float,InjectedToolArg])->float:
    '''
    this function is related to multiply the base currency or conversion rate
    ''' 
    return base_currency_value*conversion_rate

# tool binding

llm=ChatGroq(model="llama-3.3-70b-versatile" ,api_key=GROQ_KEY)

llm_with_tools=llm.bind_tools([get_conversion_factor,convert])


# tool calling

message=[HumanMessage('what is the conversion factor between the USD and INR, and based on this can you convert 10 usd to inr')]

ai_message=llm_with_tools.invoke(message)

message.append(ai_message)
 

for tool_call in ai_message. tool_calls:
    if tool_call['name']=='get_conversion_factor':
        message1=get_conversion_factor.invoke(tool_call)
        conversion_rate=json.loads(message1.content)['conversion_rate']
        message.append(message1)
    
    if tool_call['name']=='convert':
        rate=tool_call['args']['conversion_rate']=conversion_rate
        message2=convert.invoke(tool_call)
        message.append(message2)

final_result=llm_with_tools.invoke(message)

print(final_result.content)
         





 