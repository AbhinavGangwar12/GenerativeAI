from langchain_ollama import ChatOllama
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.output_parsers import StrOutputParser
from langchain_core.chat_history import InMemoryChatMessageHistory
from langchain_core.runnables.history import RunnableWithMessageHistory
from langchain_core.runnables import RunnablePassthrough

llm = ChatOllama(model="llama3")
parser = StrOutputParser()

# 1. Chain A: Dunk's Brain
prompt = ChatPromptTemplate.from_messages([
    ("system", "You are Ser Duncan the Tall. Speak humbly and mention Ser Arlan of Pennytree."),
    MessagesPlaceholder(variable_name="history"),
    ("human", "{input}") # FIXED: Added curly braces
])
dunk_chain = prompt | llm | parser

# 2. Setup Memory
store = {}
def get_history(session_id: str) -> InMemoryChatMessageHistory:
    if session_id not in store:
        store[session_id] = InMemoryChatMessageHistory()
    return store[session_id]

# 3. The Memory Wrapper (This outputs a raw string because of the parser above)
conv_chain = RunnableWithMessageHistory(
    dunk_chain,
    get_history,
    input_messages_key="input",
    history_messages_key="history"
)

# 4. Chain B: The Maester Translator
english_prompt = ChatPromptTemplate.from_template(
    "Convert the following text into highly formal, archaic Old Westerosi/Shakespearean English.\n\nText: {input_text}"
)
translation_chain = english_prompt | llm | parser

# 5. THE MASTER CHAIN (The Double Twist)
# We pipe the output of conv_chain (a string) into a dictionary, 
# mapping it to 'input_text' so the translation_chain can catch it.
master_chain = conv_chain | {"input_text": RunnablePassthrough()} | translation_chain


print("Welcome to the campfire! Type 'exit' to end the conversation.\n")
while True:
    user_input = input("You: ")
    if user_input.lower() in ["exit", "quit"]:
        print("Farewell, traveler.")
        break

    # We invoke the MASTER chain. 
    # It fetches memory, generates Dunk's response, saves to memory, AND translates it!
    res = master_chain.invoke(
        {"input" : user_input},
        config={"configurable": {"session_id" : "ashford_tourney"}}
    )

    print(f"\nSer Duncan (Translated): {res}\n")