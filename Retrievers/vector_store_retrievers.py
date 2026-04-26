from langchain_community.vectorstores import Chroma
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_core.documents import Document

model=HuggingFaceEmbeddings(
    model_name="embedingHF/fine_tuned_bilingual_model_v2",
)

 
doc1 = Document(
    page_content="""
LangChain is a framework used to build applications powered by large language models (LLMs). 
It helps developers connect LLMs with external data sources, tools, and APIs. 
LangChain is widely used for building chatbots, retrieval-augmented generation (RAG) systems, and AI agents.
""",
    metadata={"topic": "LangChain", "category": "Framework"}
)

doc2 = Document(
    page_content="""
DeepSeek is a powerful large language model designed for advanced reasoning, coding, and conversational tasks. 
It is known for its strong performance in code generation and problem-solving. 
DeepSeek models are often used as cost-effective alternatives to other LLM providers.
""",
    metadata={"topic": "DeepSeek", "category": "LLM"}
)

doc3 = Document(
    page_content="""
Artificial Intelligence (AI) is the field of computer science focused on building systems that can perform tasks requiring human intelligence. 
These tasks include decision making, natural language understanding, and image recognition. 
AI is widely used in industries like healthcare, finance, and automation.
""",
    metadata={"topic": "AI", "category": "Technology"}
)

doc4 = Document(
    page_content="""
Machine Learning (ML) is a subset of AI that enables systems to learn from data without being explicitly programmed. 
ML algorithms improve their performance as they are exposed to more data. 
Common applications include recommendation systems, fraud detection, and predictive analytics.
""",
    metadata={"topic": "Machine Learning", "category": "AI Subfield"}
)

doc5 = Document(
    page_content="""
Next.js is a popular React framework used for building modern web applications. 
It provides features like server-side rendering (SSR), static site generation (SSG), and API routes. 
Next.js helps developers build fast, scalable, and SEO-friendly applications.
""",
    metadata={"topic": "Next.js", "category": "Frontend Framework"}
)
documents=[doc1,doc2,doc3,doc4,doc5]


vector_db=Chroma.from_documents(
    documents=documents,
    embedding=model,
    collection_name='language_collection'
)

retriever=vector_db.as_retriever(search_kwargs={"k": 1})

query='what is deepseek?'

result=retriever.invoke(query)


for i in result:
    print(i.page_content)