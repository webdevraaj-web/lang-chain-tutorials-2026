from langchain_community.retrievers import WikipediaRetriever


retiever=WikipediaRetriever(
    top_k_results=2,
    lang='en'
)

query='tell me about the ipl?'

result=retiever.invoke(query)

for i in result:
    print(i.page_content)


 