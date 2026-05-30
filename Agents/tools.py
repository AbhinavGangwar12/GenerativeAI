from langchain_ollama import ChatOllama
from langchain_core.tools import tool

@tool
def get_knight_stats(knight_name: str) -> str:
    """
    Use this tool to look up the combat stats of a specific knight. 
    Pass the name of the knight as the argument.
    """

    stats = {
        "Ser Duncan the Tall": "Height: 6'11\", Weapon: Longsword, Mount: Chestnut horse named Thunder.",
        "Prince Aerion Targaryen": "Height: Average, Weapon: Lance, Armor: Bright flame enamel."
    }

    return stats.get(knight_name, "Knight not found.")

@tool 
def calculate_silver_stags(gold_dragons: int) -> int:
    """
    Use this tool to convert gold dragons to silver stags. 
    The conversion rate is 1 gold dragon = 100 silver stags.
    Pass the number of gold dragons as the argument.
    """

    return gold_dragons * 210

tools = [get_knight_stats, calculate_silver_stags]

llm = ChatOllama(model="mistral", temperature=0.0)
llm_with_tools = llm.bind_tools(tools)

# query1 = "What are the stats of Ser Duncan the Tall?"
# response1 = llm_with_tools.invoke(query1)
# print(f"Response to query 1: {response1.content}\n-----------------------------------\n")
# print("Did the LLM decide to use a tool?")
# print(response1.tool_calls, "\n")

query1 = "Ser Duncan won 5 Gold Dragons in a melee. How many Silver Stags is that?"
res = llm_with_tools.invoke(query1)
if res.tool_calls:

    print(
        f"LLM decided to use a tool! Tool name: {res.tool_calls[0]['name']}"
    )

    result = calculate_silver_stags.invoke(
        res.tool_calls[0]["args"]
    )

    print(f"Total output : {result}")
