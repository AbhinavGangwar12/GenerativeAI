from langchain_community.document_loaders import TextLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter

document = TextLoader("data.txt").load()
splitter = RecursiveCharacterTextSplitter(chunk_size=100, chunk_overlap=30, length_function=len, separators=["\n\n", "\n", " ", ""], keep_separator=False)
chunks = splitter.split_documents(document)

for i, chunk in enumerate(chunks):
    print(f"Chunk {i+1}:")
    print(chunk.page_content)
    print("-" * 40)