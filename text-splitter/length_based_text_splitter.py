from langchain_text_splitters import CharacterTextSplitter

text='''
Artificial intelligence is transforming the way humans interact with technology in everyday life. From voice assistants that can understand natural language to recommendation systems that suggest what to watch or buy, AI has become deeply integrated into modern systems. One of the most interesting areas of AI is natural language processing, where machines are trained to understand and generate human language. This allows applications such as chatbots, translation tools, and automated summarization systems to function effectively.

In recent years, large language models have gained popularity due to their ability to generate coherent and context-aware responses. These models are trained on vast amounts of text data and can perform a wide range of tasks including answering questions, writing code, and even composing creative content. However, working with large texts often requires breaking them into smaller chunks so that they can be processed efficiently. This is where text splitting techniques come into play, helping developers manage long documents and improve the performance of retrieval systems.

For example, when building a retrieval-augmented generation system, documents are first split into smaller pieces before being converted into embeddings. These embeddings are then stored in a vector database, allowing for efficient similarity search. When a user asks a question, the system retrieves the most relevant chunks and uses them to generate a response. Proper chunking ensures that important context is preserved while avoiding unnecessary noise in the data.

'''


spillter=CharacterTextSplitter(
    chunk_size=100,
    chunk_overlap=0,
    keep_separator=''
    )

result=spillter.split_text(text)
print(len(result))
print('first',result[0])
print('second',result[1])
print('third',result[2])