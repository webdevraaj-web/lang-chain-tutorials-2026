from langchain_community.document_loaders import PyPDFLoader

loader=PyPDFLoader('loaders/pdfs/langchain_30_pages.pdf')

result=loader.load()

print(result[0].page_content)
print(result[0].metadata)