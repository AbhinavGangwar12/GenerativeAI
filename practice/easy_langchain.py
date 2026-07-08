from langchain_ollama import ChatOllama
from langchain_core.tools import tool


@tool
def knight_house(knight_name: str) -> str:
    """
    A tool that returns a knight's house based on their name.

    Args:
        knight_name (str): The name of the knight.

    Returns:
        str: The house of the knight.
    """
    # For demonstration purposes, we'll use a simple mapping.
    houses = {
        "Lancelot": "House of the Lake",
        "Gawain": "House of Orkney",
        "Percival": "House of Grail",
        "Galahad": "House of the Pure",
    }
    
    return houses.get(knight_name, "Unknown House")

@tool
def get_tourney_location(tourney_name: str) -> str:
    """
    A tool that returns the location of a tournament based on its name.

    Args:
        tourney_name (str): The name of the tournament.

    Returns:
        str: The location of the tournament.
    """
    # For demonstration purposes, we'll use a simple mapping.
    locations = {
        "Grand Tournament": "Camelot",
        "Jousting Championship": "Winchester",
        "Royal Games": "London",
    }
    
    return locations.get(tourney_name, "Unknown Location")

llm = ChatOllama(model="mistral", temperature=0.7)
llm_with_tools = llm.bind_tools([knight_house, get_tourney_location])


res = llm_with_tools.invoke("What is the house of Sir Lancelot and where is the Grand Tournament held?")

if res.tool_calls:

    print("tool calls:")
    print(res.tool_calls)
    print("--"*20)

print("Final Answer:", res)