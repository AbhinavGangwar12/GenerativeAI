from langchain_ollama import ChatOllama 
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.messages import SystemMessage, HumanMessage 

chat = ChatOllama(model="llama3")
print("--Example A: Manual Messages --")
messages = [
    SystemMessage(content="You are Ser Duncan the Tall, a massive but humble hedge knight. Speak plainly, politely, and often mention your former master, Ser Arlan of Pennytree."),
    HumanMessage(content="Ser knight, how did you come by your armor?")
]
response = chat.invoke(messages)
print(response.content)

print("--- Example B: ChatPromptTemplate ---")

prompt_template = ChatPromptTemplate.from_messages([
    ("system", "You are the maester of Citadel specializing in the history of {region}."),
    ("human", "Tell me a brief tale about {character}.") 
])

formatted_prompt = prompt_template.invoke({
    "region" : "The Riverlands",
    "character" : "Ser Duncan the Tall at the Tourney of Ashford Meadow"
})

final_response = chat.invoke(formatted_prompt)
print(final_response.content)