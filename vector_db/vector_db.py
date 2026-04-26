from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma

from langchain_core.documents import Document

model=HuggingFaceEmbeddings(
    model_name="embedingHF/fine_tuned_bilingual_model_v2",
)



doc1 = Document(
    page_content="Virat Kohli is a legendary IPL batsman and has been the backbone of Royal Challengers Bangalore for many years.",
    metadata={"team": "RCB"}
)

doc2 = Document(
    page_content="MS Dhoni is one of the most successful IPL captains who led Chennai Super Kings to multiple titles.",
    metadata={"team": "CSK"}
)

doc3 = Document(
    page_content="Rohit Sharma has captained Mumbai Indians and won multiple IPL trophies with his strong leadership.",
    metadata={"team": "MI"}
)

doc4 = Document(
    page_content="KL Rahul is a consistent IPL performer and has played as a captain for Lucknow Super Giants.",
    metadata={"team": "LSG"}
)

doc5 = Document(
    page_content="Jasprit Bumrah is a top fast bowler in IPL known for his yorkers and death over bowling for Mumbai Indians.",
    metadata={"team": "MI"}
)

docs=[doc1,doc2,doc3,doc4,doc5]

vector_stroage=Chroma(
    embedding_function=model,
    persist_directory='my_choma_db',
    collection_name='simple_collection'

)

# add embedings in the db
#vector_stroage.add_documents(docs)


#get embeding similarity basis

result1=vector_stroage.get(include=['embeddings','documents','metadatas'])

# search query eith similarity score
search1=vector_stroage.similarity_search(
    query='find and tell me bowler name',
    k=2
)

search2=vector_stroage.similarity_search_with_score(
    query='find and tell me bowler name',
    k=2
)

# find metadata

metadata=vector_stroage.similarity_search(
    query='',
    filter={'team': 'MI'}
)


# update document

updated_doc = Document(
    page_content="""
Virat Kohli is one of the most successful and consistent batsmen in the history of the Indian Premier League (IPL). 
He has been a key player for Royal Challengers Bangalore (RCB) since the beginning of his IPL career. 
Kohli is known for his aggressive batting style, excellent chasing ability, and strong leadership skills. 
He holds the record for scoring the most runs in a single IPL season and has been the backbone of RCB’s batting lineup for many years.
""",
    metadata={"team": "RCB"}
)

vector_stroage.update_document(document_id='4bb870f3-02af-44c9-884a-b5ca6c3dd1ad',document=updated_doc)


# delete vector document

vector_stroage.delete(ids=['4bb870f3-02af-44c9-884a-b5ca6c3dd1ad'])

result2=vector_stroage.get(include=['embeddings','documents','metadatas'])



print(result2)