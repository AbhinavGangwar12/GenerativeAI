from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma
from langchain_ollama import ChatOllama
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough 
from langchain_community.document_loaders import TextLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter

file_path = "ashford_tourney.txt"
loader = TextLoader(file_path)
docs = loader.load()
splitter = RecursiveCharacterTextSplitter(
    chunk_size=50,
    chunk_overlap=30,
    separators=["."],
    keep_separator=False,
    length_function=len,
    is_separator_regex=False
)
chunks = splitter.split_documents(docs)

print("Loading the embedding model... ")
emb_model = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
vectorstore = Chroma.from_documents(
    documents=chunks, embedding=emb_model, collection_name="ashford_tourney"
)

retriever = vectorstore.as_retriever(search_kwargs={"k": 2})

# this function takes the list of retrieved documents and formats them into a single string that can be fed into the LLM prompt
def format_docs(docs):
    return "\n\n".join([doc.page_content for doc in docs])

template = """
You are a Maester of the Citadel. Answer the question based ONLY on the following context. 
If you do not know the answer based on the context, admit that the archives are incomplete.

Context:
{context}

Question: {question}
"""

prompt = ChatPromptTemplate.from_template(template)

llm = ChatOllama(model="llama3", temperature=0.0)
parser = StrOutputParser()

rag_chain = (
    {"context" : retriever | format_docs, "question": RunnablePassthrough()}
    | prompt
    | llm
    | parser
)

query = "Why did Ser Duncan attack a member of the royal family?"
print(f"Query: {query}\n")
res = rag_chain.invoke(query)
print(f"Answer: {res}\n")