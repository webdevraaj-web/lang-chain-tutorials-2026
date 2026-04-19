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


template=PromptTemplate(
    template='tel me important 5 points on this {topic}',
    input_variables=['topic']
)

parser=StrOutputParser()

 

chain=template | model | parser

result=chain.invoke({'topic':'deep learning'})

print(result)

chain.get_graph().print_ascii()