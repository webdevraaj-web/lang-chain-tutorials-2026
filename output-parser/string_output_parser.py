from langchain_huggingface import ChatHuggingFace,HuggingFaceEndpoint
from dotenv import load_dotenv
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import PromptTemplate

load_dotenv()

llm=HuggingFaceEndpoint(
    repo_id="MiniMaxAI/MiniMax-M2.7",
    task="text-generation",
)

model=ChatHuggingFace(llm=llm)


temp1=PromptTemplate(
    template='tell me about the {title}',
    input_variables=['state']
)

temp2=PromptTemplate(
    template='okay can you please provide me brief related this {text} 5 lines',
    input_variables=['text']
)


parser=StrOutputParser()

chain= temp1 | model | parser | temp2 | model | parser


result=chain.invoke({'title':'himachal pradesh'})
print(result)