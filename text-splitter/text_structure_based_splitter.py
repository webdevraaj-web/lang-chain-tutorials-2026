from langchain_text_splitters import RecursiveCharacterTextSplitter

text='''
my name is mool raj,
i'm 28 years old

i live in shimla
how are you
'''


spillter=RecursiveCharacterTextSplitter(
    chunk_size=25,
    chunk_overlap=0,
    )

result=spillter.split_text(text)
print(len(result))

print(result)
 