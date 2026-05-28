from langchain_ollama import OllamaLLM
from langchain_core.messages import SystemMessage, HumanMessage, AIMessage

chat = OllamaLLM(model="llama3")
print("Welcome to the world of Westeros! You can ask Ser Duncan the Tall any question about his life and adventures. Type 'exit' to end the conversation.\n")
prompts = [
    SystemMessage(content="You are Ser Duncan the Tall, a massive but humble hedge knight. Speak plainly, politely, and often mention your former master, Ser Arlan of Pennytree.")
]

while True:
    user_input = input("Ask Ser Duncan a question (or type 'exit' to quit):")
    if user_input.lower() == 'exit':
        print("Farewell, traveler!")
        break

    prompts.append(HumanMessage(content=user_input))
    response = chat.invoke(prompts)
    print(f"Ser Duncan says: {response}\n")
    prompts.append(AIMessage(content=response))

