from langchain_ollama import ChatOllama
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import TextLoader
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma 
from langchain_core.runnables.history import RunnableWithMessageHistory
from langchain_core.chat_history import InMemoryChatMessageHistory
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.messages import SystemMessage, HumanMessage
from langchain_core.tools import tool

file_path = "ashford_tourney.txt"
loader = TextLoader(file_path)
docs = loader.load()
splitter = RecursiveCharacterTextSplitter(chunk_size=50, chunk_overlap=30, separators=["."], keep_separator=False, length_function=len, is_separator_regex=False)

chunks = splitter.split_documents(docs)

embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
vectorstore = Chroma.from_documents(collection_name="ashford_tourney",documents=chunks, embedding=embeddings)
retriever = vectorstore.as_retriever(search_kwargs={"k": 2})

for i, chunk in enumerate(chunks):
    print(f"Chunk {i}: {chunk.page_content}")
    print("--"*20)

@tool
def search_historical_archives(query: str) -> str:
    """
    A tool that searches the historical archives for information related to the query.

    Args:
        query (str): The search query."""
    result = retriever.invoke(query)
    docs = " ".join([doc.page_content for doc in result])
    return docs if docs else "No relevant information found in the historical archives."

llm = ChatOllama(model="mistral", temperature=0.7)
llm_with_tools = llm.bind_tools([search_historical_archives])

prompt = ChatPromptTemplate.from_messages([
    SystemMessage(content="You are the royal squire Egg. You are a helpful assistant that knows about the knights of the round table and their tournaments."),
    MessagesPlaceholder(variable_name="hist"),
    HumanMessage(content="{input}")
])

chain = prompt | llm_with_tools 

store = {}

def get_session_history(session_id: str) -> InMemoryChatMessageHistory:
    if session_id not in store:
        store[session_id] = InMemoryChatMessageHistory()
    return store[session_id]

main_chain = RunnableWithMessageHistory(
    chain,
    get_session_history,
    input_messages_key="input",
    history_messages_key="hist"
)

config = {"configurable" : {"session_id": "default"}}
res = main_chain.invoke({"input": "Who fought alongside Ser Duncan in the Trial of Seven?"}, config=config)
if res.tool_calls:
    print(f"--- The LLM requested to use a tool: {res.tool_calls[0]['name']} ---")
    
    # Extract the argument the LLM generated (e.g., "Trial of Seven Ser Duncan")
    generated_query = res.tool_calls[0]['args']['query']
    print(f"LLM's Search Query: '{generated_query}'\n")
    
    # 4. MANUALLY run the Python function
    tool_result = search_historical_archives.invoke({"query": generated_query})
    
    print("--- Tool Output (What we would send back to the LLM) ---")
    print(tool_result)
else:
    print("Final Answer: ", res.content)

