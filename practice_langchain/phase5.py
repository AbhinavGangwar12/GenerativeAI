from langchain_ollama import ChatOllama
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_community.document_loaders import TextLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter

document = TextLoader("data.txt").load()
splitter = RecursiveCharacterTextSplitter(chunk_size=100, chunk_overlap=30, length_function=len, separators=["\n\n", "\n", " ", ""], keep_separator=False)
chunks = splitter.split_documents(document)

model = ChatOllama(model="llama3.1")
embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
vectorstore = FAISS.from_documents(chunks, embeddings)

# response = vectorstore.similarity_search("Where did Dunk and Egg compete?", k=2)

prompt = ChatPromptTemplate.from_messages([
    ("system", "Your task is to give the response to the user input based on the context provided and the question. The context is as follows: {context}\n The question is: {question}"),
])

def get_context(query: str) -> str:
    response = vectorstore.similarity_search(query, k=2)
    context = "\n".join([chunk.page_content for chunk in response])
    return context

chain = (
    {"context" : get_context , "question" : RunnablePassthrough()} |
    prompt |
    model |
    StrOutputParser()
)

print("Chain invoked.")
print(chain.invoke("Who was Egg secretly, and what rank did Dunk eventually achieve?"))