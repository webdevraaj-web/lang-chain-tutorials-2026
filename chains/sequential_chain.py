from langchain_huggingface import ChatHuggingFace,HuggingFaceEndpoint
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import PromptTemplate
from dotenv import load_dotenv

load_dotenv()


llm=HuggingFaceEndpoint(
    repo_id="MiniMaxAI/MiniMax-M2.7",
    task="text-generation"
)

model=ChatHuggingFace(llm=llm)


template1=PromptTemplate(
    template='tell me the breif overview on this {topic}',
    input_variables=['topic']
)


template2=PromptTemplate(
    template='can you provide me 5 top points on this {text}',
    input_variables=['text']
)

parser=StrOutputParser()

chain=template1 | model | parser | template2 | model | parser 

result=chain.invoke({'topic':'langchain'})

print(result)

chain.get_graph().print_ascii()