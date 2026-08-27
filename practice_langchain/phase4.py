from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_community.document_loaders import TextLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter

document = TextLoader("data.txt").load()
splitter = RecursiveCharacterTextSplitter(chunk_size=100, chunk_overlap=30, length_function=len, separators=["\n\n", "\n", " ", ""], keep_separator=False)
chunks = splitter.split_documents(document)

embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
vectorstore = FAISS.from_documents(chunks, embeddings)

response = vectorstore.similarity_search("Where did Dunk and Egg compete?", k=2)
print("Top 2 most similar chunks:")
for i, chunk in enumerate(response):
    print(f"Chunk {i+1}:")
    print(chunk.page_content)
    print("-" * 40)