from langchain_huggingface import ChatHuggingFace,HuggingFaceEndpoint
from dotenv import load_dotenv
from langchain_core.prompts import PromptTemplate

load_dotenv()

llm=HuggingFaceEndpoint(
    repo_id="MiniMaxAI/MiniMax-M2.7",
    task="text-generation",
)

model=ChatHuggingFace(llm=llm)


# prompt 1

temp1=PromptTemplate(
    template="tell me about this tpoic {topic}",
    input_variables=['topic']
)


# prompt 2

temp2=PromptTemplate(
    template="i need breif 5 line text related this {text} once",
    input_variables=['text']

)


prompt1=temp1.invoke({'topic':'langchain'})

result1=model.invoke(prompt1)


prompt2=temp2.invoke({'text':result1.content})

result2=model.invoke(prompt2)


print('result-1 :',result1.content)
print('result-2 :',result2.content)


