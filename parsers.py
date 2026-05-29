from langchain_ollama import ChatOllama 
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import JsonOutputParser, PydanticOutputParser
from langchain_core.runnables import RunnableLambda, RunnablePassthrough
from pydantic import BaseModel, Field

llm = ChatOllama(model="llama3")

print("--- Example A: Using a JSON Output Parser with a Pydantic Schema ---")
class KnightInfo(BaseModel):
    name: str = Field(description="The name of the knight")
    house: str = Field(description="The house the knight belongs to")
    weapon: str = Field(description="The knight's weapon of choice")
    honor_rating: int = Field(description="The knight's honor rating on a scale of 1 to 10")

parser = PydanticOutputParser(pydantic_object=KnightInfo)
prompt = ChatPromptTemplate.from_template(
    "Analyze {character}. \n {formatting_instructions}"
)

json_chain = prompt | llm | parser
res = json_chain.invoke(
    {
        "character": "Ser Duncan the Tall.",
        "formatting_instructions" : parser.get_format_instructions()
    }
)

print(type(res))
print(res)

print("--- Example B: Using a RunnableLambda to Post-Process the Parsed Output ---")

def honor_knight(knight_info: KnightInfo) -> str:
    """A simple function that takes the output from the parser and returns a string evaluating the knight's honor."""
    rating = knight_info.honor_rating
    name = knight_info.name
    if rating >= 8:
        return f"{name} is an honorable knight with a rating of {rating}."
    elif rating >= 5:
        return f"{name} is a somewhat honorable knight with a rating of {rating}."
    else:
        return f"{name} is not an honorable knight with a rating of {rating}."

custom_logic = RunnableLambda(honor_knight)

chain = (
    {"character": lambda x: x["character"], "formatting_instructions": lambda _:parser.get_format_instructions()}
    | prompt
    | llm
    | parser 
    | custom_logic
)

final_verdict =  chain.invoke({"character": "Ser Duncan the Tall."})
print(final_verdict)