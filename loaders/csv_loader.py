from langchain_community.document_loaders import CSVLoader

loader=CSVLoader('loaders/files/users_100_rows.csv')

result=loader.load()

print(result[0].page_content)
print(result[0].metadata)