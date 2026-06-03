from langchain_ollama import ChatOllama
from langgraph.graph import StateGraph, START, END
from typing import TypedDict, Annotated
from langchain_core.messages import BaseMessage, HumanMessage
from langgraph.checkpoint.sqlite import SqliteSaver
from langgraph.graph.message import add_messages
import sqlite3

llm = ChatOllama(model="mistral", temperature=0.9)

class ChatState(TypedDict):
    messages: Annotated[list[BaseMessage], add_messages]

def chat_node(state: ChatState):
    messages = state['messages']
    response = llm.invoke(messages)
    return {"messages": [response]}


# creating a SQLite DB
conn = sqlite3.connect(database="chatbot.db", check_same_thread=False)


checkpointer = SqliteSaver(conn=conn)
graph = StateGraph(ChatState)

graph.add_node("chat_node", chat_node)

graph.add_edge(START, "chat_node")
graph.add_edge("chat_node", END)

model = graph.compile(checkpointer=checkpointer)

#extracting total number of threads in the DB

def get_all_threads():
    threads = set()
    for checkpoint in checkpointer.list(None):
        threads.add(checkpoint.config['configurable']['thread_id'])
    return list(threads)

