from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_groq import ChatGroq
from langchain_community.vectorstores import FAISS
from langchain_core.prompts import PromptTemplate
from youtube_transcript_api import YouTubeTranscriptApi,TranscriptsDisabled
from langchain_huggingface import HuggingFaceEmbeddings
import os
from dotenv import load_dotenv
from langchain_core.runnables import RunnableLambda,RunnableParallel,RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser

load_dotenv()

# step 1  indexing:

#(a) load document

video_id='THL1OPn72vo'


try:
    api=YouTubeTranscriptApi()
    transcript_list=api.fetch(video_id,languages=['en','hi'])
    transcript=" ".join(chunk.text for chunk in transcript_list)
except TranscriptsDisabled:
    print('not video id found here')


# (b) text splitting

splitter=RecursiveCharacterTextSplitter(
    chunk_size=1000,
    chunk_overlap=100
)

chunks=splitter.create_documents([transcript])
 

# (c) created embeddings and store all embedding into the vector stroage 

embedding=HuggingFaceEmbeddings(
    model_name="embedingHF/fine_tuned_bilingual_model_v2",
)
vector_store=FAISS.from_documents(chunks,embedding)

# print(vector_store.index_to_docstore_id)


# step 2 Retrievel

# (a)

retriever=vector_store.as_retriever(search_type="mmr", search_kwargs={"k": 6})

# step 3 augmentation
 
GROQ_KEY=os.getenv('GROQ_API_KEY')
model=ChatGroq(model="llama-3.3-70b-versatile" ,api_key=GROQ_KEY)

prompt=PromptTemplate(
    template='''
    you are a helpful assistant.
    answr only the provided transcript content,
    if the context is insufficient just say you dont know
    {context}
    question:{question}
    ''',
    input_variables=['context','question']
)

# question='is the topic of aliens discuss in this video? if yes than whats was the diccussed please explain'

# retriever_doc=retriever.invoke(question)

def format_docx(retriever_doc):
    content_text="\n\n".join(i.page_content for i in retriever_doc)
    return content_text

runable_parallel=RunnableParallel({
    'context':retriever | RunnableLambda(format_docx),
    'question':RunnablePassthrough()
}
)

parser=StrOutputParser() 

final_chain=runable_parallel | prompt | model | parser


# generation step 4
query='explain me what this video related for?'

result=final_chain.invoke(query)

print(result)


 




 