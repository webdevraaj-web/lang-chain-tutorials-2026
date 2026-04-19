from langchain_huggingface import ChatHuggingFace,HuggingFaceEndpoint
from dotenv import load_dotenv
from langchain_core.output_parsers import PydanticOutputParser
from langchain_core.prompts import PromptTemplate
from pydantic import BaseModel,Field

load_dotenv()

llm=HuggingFaceEndpoint(
    repo_id="MiniMaxAI/MiniMax-M2.7",
    task="text-generation",
)

model=ChatHuggingFace(llm=llm)


class person(BaseModel):
    name: str =Field(description='person name is required')
    age: int = Field(gt=20,description='preson age is required')
    city: str=Field(description='provide the city name where the person belongs to')

parser=PydanticOutputParser(pydantic_object=person)


temp1=PromptTemplate(
    template='tell me the  name ,age city name related {country} \n {format}',
    input_variables=['country'],
    partial_variables={'format':parser.get_format_instructions()}
)

chain=temp1 | model | parser

result=chain.invoke({'country':'us'})

 
print(result)