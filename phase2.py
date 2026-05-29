from langchain_ollama import ChatOllama 
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables.history import RunnableWithMessageHistory
from langchain_core.chat_history import InMemoryChatMessageHistory

llm = ChatOllama(model="llama3")
parser = StrOutputParser()
prompt = ChatPromptTemplate.from_messages([
    ("system", "You are Ser Duncan the Tall. Speak humbly and mention Ser Arlan."),
    MessagesPlaceholder(variable_name="history"),
    ("human", "input")
])

chain = prompt | llm | parser

store = {}

def get_session_history(sesion_id: str) -> InMemoryChatMessageHistory:
    if sesion_id not in store:
        store[sesion_id] = InMemoryChatMessageHistory()
    return store[sesion_id]

conversational_chain = RunnableWithMessageHistory(
    chain,
    get_session_history,
    input_messages_key="input",
    history_messages_key="history",
)

print("--- Modern Conversation with Message History ---")

print("Turn 1 (Human): Greetings! I have a fine chestnut horse named Thunder.")
res1 = conversational_chain.invoke(
    {"input": "Greetings! I have a fine chestnut horse named Thunder."},
    config={"configurable": {"session_id": "session_1"}}
)
print(f"Turn 1 (Duncan): {res1}\n")

print("Turn 2 (Human): What was the name of my horse again?")
res2 = conversational_chain.invoke(
    {"input": "What was the name of my horse again?"},
    config={"configurable": {"session_id": "session_1"}}
)
print(f"Turn 2 (Duncan): {res2}\n")