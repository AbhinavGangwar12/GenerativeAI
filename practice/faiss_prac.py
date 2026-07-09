# import os
# import faiss
# import numpy as np
# from langchain_ollama import ChatOllama
# from langchain_huggingface import HuggingFaceEmbeddings
# from langchain_text_splitters import RecursiveCharacterTextSplitter

# from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
# from langchain_core.messages import SystemMessage, HumanMessage
# from langchain_core.runnables.history import RunnableWithMessageHistory
# from langchain_core.chat_history import InMemoryChatMessageHistory
# from langchain_core.tools import tool

# print("Agent booting...")

# file = "ashford_tourney.txt"
# vector_store_dir = "vector_store"
# index_file = os.path.join(vector_store_dir, "vector_index.index")

# GLOBAL_DOCS = []
# GLOBAL_INDEX = None

# embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")


# def load_vector_store():
#     global GLOBAL_DOCS, GLOBAL_INDEX

#     documents = open(file, "r").read()
#     splitter = RecursiveCharacterTextSplitter(chunk_size=30, chunk_overlap=0, separators=["."], is_separator_regex=False, keep_separator=False)
#     chunks = splitter.split_text(documents)
#     for chunk in chunks:
#         GLOBAL_DOCS.append(chunk)
    
#     if os.path.exists(index_file):
#         print("Loading existing index from file...")
#         GLOBAL_INDEX = faiss.read_index(index_file)
#         return 
    
#     print("Creating new index...")

#     vectors = embeddings.embed_documents(GLOBAL_DOCS)
#     vectors = np.array(vectors, dtype=np.float32)
#     dim = vectors.shape[1]

#     GLOBAL_INDEX = faiss.IndexFlatL2(dim)
#     GLOBAL_INDEX.add(vectors)

#     faiss.write_index(GLOBAL_INDEX, index_file)
#     return 
# load_vector_store()

# @tool
# def search_tourney_texts(query: str, k: int = 3) -> list[str]:
#     """
#     A tool that searches the historical archives for information related to the query.

#     Args:
#         query (str): The search query."""
#     global GLOBAL_DOCS, GLOBAL_INDEX
#     query_vector = embeddings.embed_query(query)
#     query_vector = np.array([query_vector], dtype=np.float32)
#     _, I = GLOBAL_INDEX.search(query_vector, k)
#     results = [GLOBAL_DOCS[i] for i in I[0]]
#     return results

# llm = ChatOllama(model="mistral", temperature=0.7)
# llm_with_tools = llm.bind_tools([search_tourney_texts])


# prompt = ChatPromptTemplate.from_messages([
#     SystemMessage(content="You are the royal squire Egg. You are a helpful assistant that knows about the knights of the round table and their tournaments."),
#     MessagesPlaceholder(variable_name="hist"),
#     HumanMessage(content="{input}")
# ])

# chain = prompt | llm_with_tools

# store = {}

# def get_session_id(token: str) -> str:
#     if token not in store:
#         store[token] = InMemoryChatMessageHistory()
#     return store[token]

# main_chain = RunnableWithMessageHistory(
#     chain,
#     get_session_id,
#     input_messages_key="input",
#     history_messages_key="hist"
# )

# config = {"configurable" : {"session_id" : "temp-01"}}
# query = "Who fought alongside Ser Duncan in the Trial of Seven?"

# res = main_chain.invoke({"input": query}, config = config)
# if res.tool_calls:
#     print(f"--- The LLM requested to use a tool: {res.tool_calls[0]['name']} ---")
#     generated_query = res.tool_calls[0]['args']['query']
#     print(f"--- The LLM generated the following query for the tool: {generated_query} ---")

#     tool_results = search_tourney_texts.invoke({"query": generated_query, "k": 3})
#     print(f"--- The tool returned the following results: {tool_results} ---")
#     print(tool_results)
# else:
#     print(f"final answer: {res.content}")


# a better way to do this using langgraph react agent 
import os
import faiss
import numpy as np
from langchain_ollama import ChatOllama
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.tools import tool

# Import LangGraph components
from langgraph.checkpoint.memory import MemorySaver
from langgraph.prebuilt import create_react_agent

print("Agent booting...")

file = "ashford_tourney.txt"
vector_store_dir = "vector_store"
index_file = os.path.join(vector_store_dir, "vector_index.index")

GLOBAL_DOCS = []
GLOBAL_INDEX = None

embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")

def load_vector_store():
    global GLOBAL_DOCS, GLOBAL_INDEX
    # (Your existing vector store loading code stays exactly the same)
    if not os.path.exists(file):
        # Dummy data for demonstration if file doesn't exist
        GLOBAL_DOCS = ["Ser Duncan fought alongside Egan Estermont, Ser Allyn, and Prince Baelor."]
    else:
        documents = open(file, "r").read()
        splitter = RecursiveCharacterTextSplitter(chunk_size=100, chunk_overlap=0, separators=["."], keep_separator=False)
        GLOBAL_DOCS = splitter.split_text(documents)
    
    if os.path.exists(index_file):
        GLOBAL_INDEX = faiss.read_index(index_file)
        return 
    
    vectors = embeddings.embed_documents(GLOBAL_DOCS)
    vectors = np.array(vectors, dtype=np.float32)
    dim = vectors.shape[1]
    GLOBAL_INDEX = faiss.IndexFlatL2(dim)
    GLOBAL_INDEX.add(vectors)
    faiss.write_index(GLOBAL_INDEX, index_file)

load_vector_store()

@tool
def search_tourney_texts(query: str) -> str:
    """A tool that searches the historical archives for information related to the query."""
    global GLOBAL_DOCS, GLOBAL_INDEX
    query_vector = embeddings.embed_query(query)
    query_vector = np.array([query_vector], dtype=np.float32)
    _, I = GLOBAL_INDEX.search(query_vector, k=2)
    results = [GLOBAL_DOCS[i] for i in I[0] if i < len(GLOBAL_DOCS)]
    return "\n".join(results)

# 1. Define your tools list
tools = [search_tourney_texts]

# 2. Use a model variant that handles tools well, or prompt it explicitly
llm = ChatOllama(model="mistral", temperature=0)

# 3. Define a clear system prompt reminding the model it HAS tools
system_prompt = (
    "You are the royal squire Egg. You are a helpful assistant that knows about the knights "
    "of the round table and their tournaments. You have access to tools to search historical archives. "
    "If you do not know the answer, you MUST use the search_tourney_texts tool before answering."
)

# 4. Initialize LangGraph's memory saver (replaces RunnableWithMessageHistory)
memory = MemorySaver()

# 5. Create the compiled ReAct agent
agent_executor = create_react_agent(
    llm, 
    tools, 
    checkpointer=memory,
    prompt=system_prompt
)

# 6. Run the agent
config = {"configurable": {"thread_id": "temp-01"}}
query = "Who fought alongside Ser Duncan in the Trial of Seven?"

# LangGraph manages the loop: LLM -> Tool -> LLM -> Final Answer
events = agent_executor.stream({"messages": [("user", query)]}, config=config)

for event in events:
    for node, value in event.items():
        print(f"\n--- Node: {node} ---")
        if "messages" in value:
            last_msg = value["messages"][-1]
            last_msg.pretty_print()

    