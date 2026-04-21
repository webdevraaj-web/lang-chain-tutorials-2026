from langchain_community.document_loaders import DirectoryLoader, PyPDFLoader

loader=DirectoryLoader(
    path='loaders/pdfs',
    glob="*.pdf",
    loader_cls=PyPDFLoader
)

result=loader.load()

# print(result[0].page_content)
# print(result[0].metadata)

for i in result:
    print(i)