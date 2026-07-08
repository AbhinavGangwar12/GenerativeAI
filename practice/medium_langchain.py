from langchain_ollama import ChatOllama
from langchain_core.messages import SystemMessage, HumanMessage
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.runnables.history import RunnableWithMessageHistory
from langchain_core.chat_history import InMemoryChatMessageHistory
from langchain_core.output_parsers import StrOutputParser


llm = ChatOllama(model="mistral", temperature=0.7)
parser = StrOutputParser()
prompt = ChatPromptTemplate.from_messages([
    SystemMessage(content="You are the royal squire Egg. You are a helpful assistant that knows about the knights of the round table and their tournaments."),
    MessagesPlaceholder(variable_name="hist"),
    HumanMessage(content="{input}"),
])

chain = prompt | llm | parser

store = {}

def get_session_history(session_id: str) -> InMemoryChatMessageHistory:
    if session_id not in store:
        store[session_id] = InMemoryChatMessageHistory()
    return store[session_id]

main_chain = RunnableWithMessageHistory(
    chain,
    get_session_history,
    input_messages_key="input",
    history_messages_key="hist"
)

config = {"configurable" : {"session_id": "default"}}
res1 = main_chain.invoke({"input": "What is the house of Sir Lancelot and where is the Grand Tournament held?"}, config=config)
print("Turn 1 : ", res1)

