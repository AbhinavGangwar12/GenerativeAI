import asyncio
import os
from typing import Annotated, TypedDict

import dotenv
import requests
from langchain_community.tools import DuckDuckGoSearchRun
from langchain_core.messages import BaseMessage, HumanMessage
from langchain_core.tools import tool
from langchain_ollama import ChatOllama
from langgraph.checkpoint.sqlite.aio import AsyncSqliteSaver  # <--- Async Checkpointer
from langgraph.graph import END, START, StateGraph
from langgraph.graph.message import add_messages
from langgraph.prebuilt import ToolNode, tools_condition
from langchain_mcp_adapters.client import MultiServerMCPClient

# Load environment variables
dotenv.load_dotenv()

# Initialize the LLM (Mistral via Ollama)
llm = ChatOllama(model="mistral", temperature=0.9)


# Define State Schema
class ChatState(TypedDict):
    messages: Annotated[list[BaseMessage], add_messages]


# --- TOOL DEFINITIONS ---

client = MultiServerMCPClient(
    {
        "tools" : {
            "transport" : "stdio",
            "command": "python",
            "args" : ["/home/pinaka-linux/langchain/ChatBot/mcp_server.py"]
        }
    }
)


# --- GRAPH NODE DEFINITIONS ---




async def create_graph(checkpointer):

    tools = await client.get_tools()
    llm_with_tools = llm.bind_tools(tools)

    #creating the nodes
    tool_node = ToolNode(tools=tools)
    async def chat_node(state: ChatState):
        messages = state["messages"]
        response = await llm_with_tools.ainvoke(messages)
        return {"messages": [response]}

    graph = StateGraph(ChatState)

    # Add nodes
    graph.add_node("chat_node", chat_node)
    graph.add_node("tools", tool_node)

    # Add edges
    graph.add_edge(START, "chat_node")
    graph.add_conditional_edges("chat_node", tools_condition)
    graph.add_edge("tools", "chat_node")

    # Compile with checkpointer
    return graph.compile(checkpointer=checkpointer)


# --- MAIN EXECUTION ---


async def main():
    # Use AsyncSqliteSaver for async execution (ainvoke)
    # This automatically handles DB setup via a connection string
    async with AsyncSqliteSaver.from_conn_string("chatbot.db") as checkpointer:

        # Build the model
        model = await create_graph(checkpointer)

        # Example invocation
        config = {"configurable": {"thread_id": "test_thread"}}
        initial_state = {"messages": [HumanMessage(content="Hello, my name is John.")]}

        print("--- Sending Message ---")
        res = await model.ainvoke(initial_state, config=config)

        # Print out the final message response cleanly
        print("\n--- Model Response ---")
        print(res["messages"][-1].content)


if __name__ == "__main__":
    asyncio.run(main())