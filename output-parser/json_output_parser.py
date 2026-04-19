from langchain_huggingface import ChatHuggingFace,HuggingFaceEndpoint
from dotenv import load_dotenv
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import JsonOutputParser

load_dotenv()

llm=HuggingFaceEndpoint(
    repo_id="MiniMaxAI/MiniMax-M2.7",
    task="text-generation",
)

model=ChatHuggingFace(llm=llm)

parser=JsonOutputParser()

temp1=PromptTemplate(
    template='okay give me 5 facts this {country} \n {format}',
    input_variables=['country'],
    partial_variables={'format':parser.get_format_instructions()}
)

chain= temp1 | model | parser


result=chain.invoke({'country':'deep learing'})
 

print(result)



