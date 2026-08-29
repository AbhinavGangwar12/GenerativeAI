import os
import faiss
import numpy as np
import pickle 
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_ollama import ChatOllama
from langchain_text_splitters import RecursiveCharacterTextSplitter 
from langchain_core.tools import tool
from langchain_core.runnables.history import RunnableWithMessageHistory
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_community.chat_message_histories import ChatMessageHistory
from langchain_classic.agents import create_tool_calling_agent, AgentExecutor


GLOBAL_INDEX = None
GLOBAL_DOCS = None

INDEX_PATH = "grand_faiss_index.bin"
DOCSTORE_PATH = "grand_docstore.pkl"
DOC_PATH = "grand_data.txt"
embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
llm = ChatOllama(model="llama3.1")
store = {}

def get_history(session_id: str) -> ChatMessageHistory:
    if session_id not in store:
        store[session_id] = ChatMessageHistory()
    return store[session_id]


def create_faiss_index():
    global GLOBAL_INDEX, GLOBAL_DOCS

    if os.path.exists(INDEX_PATH) and os.path.exists(DOCSTORE_PATH):
        print(f"Loading existing FAISS index and docstore...")
        GLOBAL_INDEX = faiss.read_index(INDEX_PATH)
        with open(DOCSTORE_PATH, "rb") as f:
            GLOBAL_DOCS = pickle.load(f)
        return

    print("Building new index from scratch...")
    if os.path.exists(DOC_PATH):
        documents = open(DOC_PATH, "r").read()
        text_splitter = RecursiveCharacterTextSplitter(chunk_size=100, chunk_overlap=20, separators=["\n\n", "\n", " ", ""], length_function=len, keep_separator=False)
        docs = text_splitter.split_text(documents)

        vectors = embeddings.embed_documents(docs)
        vectors = np.array(vectors, dtype=np.float32)
        dim = vectors.shape[1]

        GLOBAL_INDEX = faiss.IndexFlatL2(dim)
        GLOBAL_INDEX.add(vectors)
        GLOBAL_DOCS = docs
        faiss.write_index(GLOBAL_INDEX, INDEX_PATH)
        with open(DOCSTORE_PATH, "wb") as f:
            pickle.dump(GLOBAL_DOCS, f)
    else:
        print(f"Document file '{DOC_PATH}' not found. Please provide a valid document file.")
    return 



@tool
def search_faiss(query: str, k: int = 2) -> str:
    """Searches the FAISS index for the most relevant information on tourney in westros based on the query.
    
    Args:
        query (str): The search query.
        k (int): The number of results to return.

    Returns:
        str: The search results.
    """
    global GLOBAL_INDEX, GLOBAL_DOCS

    if GLOBAL_INDEX is None or GLOBAL_DOCS is None:
        create_faiss_index()

    query_vector = embeddings.embed_query(query)
    query_vector = np.array([query_vector], dtype=np.float32)

    _, idx = GLOBAL_INDEX.search(query_vector, k)
    return (" ").join([GLOBAL_DOCS[i] for i in idx[0]])

@tool
def calculate_winning(bouts_won: int, wager_multiplier: float) -> float:
    """Calculates the winning amount based on the number of bouts won and the wager multiplier.
    
    Args:
        bouts_won (int): The number of bouts won.
        wager_multiplier (float): The wager multiplier.

    Returns:
        float: The calculated winning amount.
    """
    return (bouts_won * 50) * wager_multiplier

prompt = ChatPromptTemplate.from_messages([
    ("system", "You are the Master of Coin and Lore for Ser Duncan the Tall. You have access to tools to calculate winnings and retrieve lore. Answer the user's questions using these tools when necessary."),
    MessagesPlaceholder(variable_name="history"),
    ("human", "{input}"),
    MessagesPlaceholder(variable_name="agent_scratchpad")
])


agent = create_tool_calling_agent(llm, [search_faiss, calculate_winning], prompt=prompt)
executor = AgentExecutor(agent=agent, tools=[search_faiss, calculate_winning],verbose=True)
master_chain = RunnableWithMessageHistory(executor, get_history, history_messages_key="history", input_messages_key="input")


config = {"configurable" : {"session_id": "user1"}}


print("\n--- TURN 1 ---")
response1 = master_chain.invoke({"input" : "Which house did the seasoned knight belong to that Dunk defeated first?"}, config=config)
# FIXED: Print ['output'] instead of .content
print(response1["output"])

print("\n--- TURN 2 ---")
response2 = master_chain.invoke({"input" : "Excellent. Dunk won 2 bouts in total at that tourney. Assuming Egg secured a wager multiplier of 1.5, use your calculator to figure out how many silver stags we won. Also, remind me of the name of the Lord who oversaw the tourney?"}, config=config)
print(response2["output"])