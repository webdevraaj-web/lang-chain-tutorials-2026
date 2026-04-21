from langchain_community.document_loaders import WebBaseLoader,TextLoader
from langchain_groq import ChatGroq
from dotenv import load_dotenv
from langchain_core.prompts import PromptTemplate
from bs4 import BeautifulSoup
from langchain_core.output_parsers import StrOutputParser
import os
load_dotenv('.env')

GROQ_KEY=os.getenv('GROQ_API_KEY')

 

model=ChatGroq(model="llama-3.3-70b-versatile" ,api_key=GROQ_KEY)


url="https://en.wikipedia.org/wiki/LangChain"

loader=WebBaseLoader(url,requests_kwargs={
        "headers": {
            "User-Agent": "Mozilla/5.0"
        }
    })


prompt=PromptTemplate(
    template='okay need to provide me mots important 5 points in details  related this {topic} form the following  {text}',
    input_variables=['topic','text']
)

parser=StrOutputParser()

chain=prompt | model | parser
result=loader.load()

data=chain.invoke({'topic':'why can we use langchain?','text':result[0].page_content})

print(data)




 
 
