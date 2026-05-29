from langchain_ollama import ChatOllama
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.output_parsers import StrOutputParser
from langchain_core.chat_history import InMemoryChatMessageHistory
from langchain_core.runnables.history import RunnableWithMessageHistory
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import TextLoader
from langchain_core.runnables import RunnablePassthrough

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

embedding = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
vectorstore = Chroma.from_documents(collection_name="ashford_tourney", embedding=embedding, documents=chunks)
retriever = vectorstore.as_retriever(search_kwargs={"k": 1})

prompt = ChatPromptTemplate.from_messages([
    ("system", "You are a Maester of the Citadel. Answer based ONLY on the following context. If unknown, admit it.\n\nContext:\n{context}"),
    MessagesPlaceholder(variable_name="history"), # History is a list of messages, so it uses the placeholder
    ("human", "{question}")
])

llm = ChatOllama(model="llama3", temperature=0.0)
parser = StrOutputParser()
store = {}

def session_history(session_id: str) -> InMemoryChatMessageHistory:
    if session_id not in store:
        store[session_id] = InMemoryChatMessageHistory()
    return store[session_id]

def format_retrieved_docs(input_dict):
    # Grab the question from the incoming dictionary
    question = input_dict["question"]
    # Retrieve the documents
    docs = retriever.invoke(question)
    # Join them into a single string
    return "\n\n".join([doc.page_content for doc in docs])

core_rag_chain = (
    RunnablePassthrough.assign(context=format_retrieved_docs) | prompt | llm | parser
)

final_conv = RunnableWithMessageHistory(
    core_rag_chain,
    session_history,
    input_messages_key="question",
    history_messages_key="history"
)

print("Welcome to the Citadel Archives! (type exit to quit)\n")
while True:
    query = input("You: ")
    if query.lower() == "exit":
        print("Farewell, and may the Seven guide you!")
        break

    # We only need to pass the "question". The wrapper handles history, and the assign() handles context!
    response = final_conv.invoke(
        {"question": query},
        config={"configurable": {"session_id": "dunk_session"}}
    )
    print(f"Maester: {response}\n")