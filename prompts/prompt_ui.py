from langchain_huggingface import ChatHuggingFace,HuggingFaceEndpoint
from dotenv import load_dotenv
from langchain_core.prompts import load_prompt
import streamlit as st
load_dotenv()

llm=HuggingFaceEndpoint(
    repo_id="MiniMaxAI/MiniMax-M2.7",
    task="text-generation",
    # max_new_tokens=100
)

model=ChatHuggingFace(llm=llm)

st.header('Research Tool')

name_query=st.selectbox("select name",["Virat Kohli","MS Dhoni"])

template = load_prompt('./template.json')

 
if st.button('Generate'):
    chain=template | model
    result=chain.invoke({
      'name_query':name_query
    })
    st.write(result.content)


