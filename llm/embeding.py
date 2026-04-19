from langchain_huggingface import HuggingFaceEmbeddings
from sklearn.metrics.pairwise import cosine_similarity
from dotenv import load_dotenv
load_dotenv()

 
model=HuggingFaceEmbeddings(
    model_name="embedingHF/fine_tuned_bilingual_model_v2",
    # if any close source used
    # dimensions=32
)

documents=[
    "Delhi is the capital of India.",
    "Mumbai is the financial capital of India.",
    "Paris is the capital of France."
    "India is a country in Asia.",
    "Berlin is the capital of Germany."
]

query="What is the capital of Germany"

embed_doc=model.embed_documents(documents)
embed_query=model.embed_query(query)


scores=cosine_similarity([embed_query],embed_doc)[0]

index,score=sorted(list(enumerate(scores)),key=lambda x:x[1])[-1]

print("query", query)
print("index", documents[index])
print("score", score)