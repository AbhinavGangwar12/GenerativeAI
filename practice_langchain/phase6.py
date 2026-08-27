# from langchain_ollama import ChatOllama
# from langchain_core.tools import tool
# from langchain.agents import create_agent

# # 1. The Tool
# @tool
# def calculate_dragon_wingspan(age_in_years: int) -> int:
#     """Calculates the wingspan of a dragon in feet based on its age."""
#     return age_in_years * 1.5 + 20

# tools = [calculate_dragon_wingspan]

# # 2. The LLM
# llm = ChatOllama(model="llama3.1")

# # 3. Create the Agent using LangGraph
# # LangGraph's prebuilt agent automatically handles the prompt, thinking, and scratchpad.
# agent = create_agent(llm, tools=tools)


# # 4. The Test
# print("Invoking the LangGraph agent...")

# # In LangGraph, we pass inputs as a list of messages rather than a raw string.
# response = agent.invoke({
#     "messages": [("user", "How wide is the wingspan of a 250-year-old dragon named Balerion?")]
# })

# # The response dict contains the full conversation history. 
# # The last message is the LLM's final synthesized answer.
# print("\n--- FINAL ANSWER ---")
# print(response["messages"][-1].content)


from langchain_ollama import ChatOllama
from langchain_core.tools import tool
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder

# Import BOTH the planner and the executor from classic
from langchain_classic.agents import create_tool_calling_agent, AgentExecutor

# 1. The Tool
@tool
def calculate_dragon_wingspan(age_in_years: int) -> int:
    """Calculates the wingspan of a dragon in feet based on its age."""
    return age_in_years * 1.5 + 20

tools = [calculate_dragon_wingspan]

# 2. The LLM
llm = ChatOllama(model="llama3.1")

# 3. The Classic Prompt
# Classic agents require you to manually define where the input goes and where the agent thinks
prompt = ChatPromptTemplate.from_messages([
    ("system", "You are a helpful assistant. Use the provided tools to answer the question."),
    ("human", "{input}"),
    MessagesPlaceholder(variable_name="agent_scratchpad"),
])

# 4. Create the Agent and Executor
# create_tool_calling_agent just creates the planner (not the executor)
agent_planner = create_tool_calling_agent(llm, tools, prompt)

# We wrap the planner in the classic executor
executor = AgentExecutor(agent=agent_planner, tools=tools, verbose=True)

print("Invoking the Classic agent...\n")

# 5. The Test (Notice classic uses 'input', not 'messages')
response = executor.invoke({
    "input": "How wide is the wingspan of a 250-year-old dragon named Balerion?"
})

print("\n--- FINAL ANSWER ---")
print(response["output"])