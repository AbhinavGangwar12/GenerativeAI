import asyncio
import os
import threading
from typing import Annotated, TypedDict
import dotenv
from langchain_core.messages import BaseMessage
from langchain_ollama import ChatOllama
from langgraph.checkpoint.sqlite.aio import AsyncSqliteSaver
from langgraph.graph import END, START, StateGraph
from langgraph.graph.message import add_messages
from langgraph.prebuilt import ToolNode, tools_condition
from langchain_mcp_adapters.client import MultiServerMCPClient

dotenv.load_dotenv()

# Background loop for running async tasks from synchronous Streamlit
_ASYNC_LOOP = asyncio.new_event_loop()
_ASYNC_THREAD = threading.Thread(target=_ASYNC_LOOP.run_forever, daemon=True)
_ASYNC_THREAD.start()

def _submit_async(coro):
    return asyncio.run_coroutine_threadsafe(coro, _ASYNC_LOOP)

def run_async(coro):
    return _submit_async(coro).result()

def submit_async_task(coro):
    return _submit_async(coro)

# Initialize LLM
llm = ChatOllama(model="mistral", temperature=0.9)

class ChatState(TypedDict):
    messages: Annotated[list[BaseMessage], add_messages]

# Global placeholders
model = None
checkpointer = None

async def init_system():
    global model, checkpointer
    # 1. Initialize DB Connection and keep it alive
    checkpointer = AsyncSqliteSaver.from_conn_string("chatbot.db")
    await checkpointer.__aenter__() # Manually enter context to keep it globally persistent
    
    # 2. Initialize MCP Client
    client = MultiServerMCPClient(
        {
            "tools": {
                "transport": "stdio",
                "command": "python",
                "args": ["/home/pinaka-linux/langchain/ChatBot/mcp_server.py"]
            }
        }
    )
    
    tools = await client.get_tools()
    llm_with_tools = llm.bind_tools(tools)
    tool_node = ToolNode(tools=tools)

    async def chat_node(state: ChatState):
        response = await llm_with_tools.ainvoke(state["messages"])
        return {"messages": [response]}

    graph = StateGraph(ChatState)
    graph.add_node("chat_node", chat_node)
    graph.add_node("tools", tool_node)
    graph.add_edge(START, "chat_node")
    graph.add_conditional_edges("chat_node", tools_condition)
    graph.add_edge("tools", "chat_node")

    model = graph.compile(checkpointer=checkpointer)

# Synchronously initialize the background state on import
run_async(init_system())

async def _alist_threads():
    threads = set()
    async for checkpoint in checkpointer.list(None):
        threads.add(checkpoint.config['configurable']['thread_id'])
    return list(threads)

def get_all_threads():
    return run_async(_alist_threads())