from langchain_huggingface import ChatHuggingFace,HuggingFaceEndpoint
from dotenv import load_dotenv
from langchain_core.prompts import PromptTemplate
from langchain_core.runnables import RunnableLambda,RunnableBranch
from langchain_core.output_parsers import StrOutputParser,PydanticOutputParser
from typing import Literal
from pydantic import BaseModel,Field

from langchain_groq import ChatGroq
import os

load_dotenv('.env')

GROQ_KEY=os.getenv('GROQ_API_KEY')

# llm=HuggingFaceEndpoint(
#     repo_id="comfyanonymous/flux_text_encoders",
#     task="text-generation"
# )

model=ChatGroq(model="llama-3.3-70b-versatile" ,api_key=GROQ_KEY)

parser=StrOutputParser()

class Sentiment(BaseModel):
    sentiment: Literal['positive','negative'] =Field(description='give the sentiment of the feedback')

parser1=PydanticOutputParser(pydantic_object=Sentiment)


prompt1=PromptTemplate(
    template='classify the sentiment based on the following {feedback} positive or negative \n {output}',
    input_variables=['text'],
    partial_variables={'output':parser1.get_format_instructions()}
)

classify_chain=prompt1 | model | parser1


prompt2=PromptTemplate(
    template='write 2 lines response to the user if sentiment is positive {feedback}',
    input_variables=['feedback']
)


prompt3=PromptTemplate(
    template='write 2 lines response to the user if sentiment is negative {feedback}',
    input_variables=['feedback']
)
 

branch_chain=RunnableBranch(
    (lambda x:x.sentiment =='positive', prompt2 | model | parser),
    (lambda x:x.sentiment =='negative', prompt3 | model | parser),
    RunnableLambda(lambda x:'no sentiment found')
) 

chain=classify_chain | branch_chain

result=chain.invoke({'feedback':'last time i have purchased a laotop from the amazon its too bad waste of money!'})

print(result)