from langchain_ollama import ChatOllama
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser

print("Model loading...")
model = ChatOllama(model="llama3.1")
print("Model loading complete.")

# Use the tuple shortcut for dynamic messages
prompt = ChatPromptTemplate.from_messages([
    ("system", "Your task is to write a short, grand and boastful introduction for the knight named {name} and his squire named {squire}. These are the people of Westeros.")
])

print("Prompt and model loaded. Invoking the chain...")
chain = prompt | model | StrOutputParser()

print("Chain invoked.")

# Bringing back Ser Duncan and Egg!
response = chain.invoke({"name": "Ser Duncan the Tall", "squire": "Egg"})
print(response)