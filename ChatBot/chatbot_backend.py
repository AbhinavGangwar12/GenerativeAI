import sqlite3
import requests
from typing import TypedDict, Annotated

from langchain_ollama import ChatOllama
from langchain_core.messages import BaseMessage
from langchain_core.tools import tool
from langchain_community.tools import DuckDuckGoSearchRun

from langgraph.graph import StateGraph, START, END
from langgraph.graph.message import add_messages
from langgraph.checkpoint.sqlite import SqliteSaver
from langgraph.prebuilt import ToolNode, tools_condition

# Initialize the LLM
llm = ChatOllama(model="mistral", temperature=0.9)

# Define State Schema
class ChatState(TypedDict):
    messages: Annotated[list[BaseMessage], add_messages]

# --- TOOL DEFINITIONS ---

search_tool = DuckDuckGoSearchRun(region='us-en')

# REMOVED @traceable here
@tool("calculator")
def calculator(first_number: float, second_number: float, operation: str) -> dict:
    """
    Perform a basic arithmetic operation on two numbers.
    Supported operations: add, sub, mul, div
    """
    try:
        if operation == "add":
            result = first_number + second_number
        elif operation == "sub":
            result = first_number - second_number
        elif operation == "mul":
            result = first_number * second_number
        elif operation == "div":
            result = first_number / second_number
        else:
            return {"error": "Unsupported operation"}
        
        return {
            "first_number": first_number, 
            "second_number": second_number, 
            "operation": operation, 
            "result": result
        }
    except Exception as e:
        return {"error": str(e)}
    
# REMOVED @traceable here
@tool("get-stock")
def get_stock_price(symbol: str) -> dict:
    """
    Fetch the current stock price for the given stock symbol using the Alpha Vantage API.
    """
    url = f"https://www.alphavantage.co/query?function=GLOBAL_QUOTE&symbol={symbol}&apikey=N67JWRVPZ21AZO3N"
    r = requests.get(url)
    return r.json()

# Bind tools to the LLM instance
tools_list = [search_tool, calculator, get_stock_price]
llm_with_tools = llm.bind_tools(tools_list)

# --- GRAPH NODE DEFINITIONS ---

# REMOVED @traceable here
def chat_node(state: ChatState):
    messages = state['messages']
    response = llm_with_tools.invoke(messages)
    return {"messages": [response]}

tool_node = ToolNode(tools_list)

# --- GRAPH COMPILATION ---
conn = sqlite3.connect(database="chatbot.db", check_same_thread=False)
checkpointer = SqliteSaver(conn=conn)

graph = StateGraph(ChatState)

graph.add_node("chat_node", chat_node)
graph.add_node("tools", tool_node)

graph.add_edge(START, "chat_node")
graph.add_conditional_edges("chat_node", tools_condition)
graph.add_edge("tools", "chat_node")  

model = graph.compile(checkpointer=checkpointer)

# --- DATABASE UTILITIES ---
def get_all_threads():
    threads = set()
    for checkpoint in checkpointer.list(None):
        threads.add(checkpoint.config['configurable']['thread_id'])
    return list(threads)