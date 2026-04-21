from langchain_community.document_loaders import TextLoader
from langchain_groq import ChatGroq
from langchain_core.output_parsers import StrOutputParser
import os
from dotenv import load_dotenv

from langchain_core.prompts import PromptTemplate

load_dotenv('.env')

GROQ_KEY=os.getenv('GROQ_API_KEY')

 

model=ChatGroq(model="llama-3.3-70b-versatile" ,api_key=GROQ_KEY)

parser=StrOutputParser()

loader=TextLoader('loaders/files/langchain_overview.txt',encoding='utf-8')




doc=loader.load()


prompt=PromptTemplate(
    template='okay i need short but impressive summary related this {topic}',
    input_variables=['topic']
)

chain=prompt | model | parser
result=chain.invoke({'topic':doc[0].page_content})
 
print(result)