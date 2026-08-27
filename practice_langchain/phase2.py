from langchain_ollama import ChatOllama
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.prompts import MessagesPlaceholder
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables.history import RunnableWithMessageHistory
from langchain_community.chat_message_histories import ChatMessageHistory
print("Modules loaded.")

model = ChatOllama(model="llama3.1")
store = {}

def get_history(key: str) -> ChatMessageHistory:
    if key not in store:
        store[key] = ChatMessageHistory()
    return store[key]

prompt = ChatPromptTemplate.from_messages([
    ("system", "Your task is to write a short, grand and boastful introduction for the knight named {name} and his squire named {squire}. These are the people of Westeros."),
    MessagesPlaceholder(variable_name="history"),
    ("human", "{user_input}")
])



llm_chain = prompt | model | StrOutputParser()
history_chain = RunnableWithMessageHistory(
    llm_chain,
    get_history,
    input_messages_key="user_input",
    history_messages_key="history"
)

print("--- TEST 1 ---")
response_1 = history_chain.invoke(
    {"name": "Ser Duncan the Tall", "squire": "Egg", "user_input": "Please introduce us."},
    config={"configurable": {"session_id": "dunk_and_egg_session"}} # Injecting the session ID
)
print(response_1)

# TEST 2: Testing the Memory
print("\n--- TEST 2 ---")
response_2 = history_chain.invoke(
    {"name": "Ser Duncan the Tall", "squire": "Egg", "user_input": "Wait, I forgot! Dunk is now the Lord Commander of the Kingsguard. Update the intro to reflect his new rank."},
    config={"configurable": {"session_id": "dunk_and_egg_session"}} # Using the SAME session ID so it remembers!
)
print(response_2)