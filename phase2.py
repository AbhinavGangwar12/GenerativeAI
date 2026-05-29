# It is about the memory components in langchain which provides the ability to store and retrieve information across interactions with the LLM. This allows for more dynamic and context-aware conversations, as the model can remember previous interactions and use that information to inform its responses. The memory components can be used to create more engaging and personalized experiences for users, as well as to enable more complex interactions with the LLM.

from langchain_ollama import OllamaLLM
from langchain_classic.chains import ConversationChain
from langchain_classic.memory import ConversationBufferMemory, ConversationSummaryMemory

llm = OllamaLLM(model="llama3")


print("--- Example A: How Buffer Memory Works Under the Hood ---")

buffer_memory = ConversationBufferMemory(return_messages=True) # this will return the list of messages in the buffer memory instead of a string summary of the conversation history

# adding the messages to the buffer memory
buffer_memory.save_context(
    {"input": "What is your name?"}, 
    {"output": "My name is Ser Duncan the Tall, a hedge knight."}
)
print("Buffer Memory Messages: ")
print(buffer_memory.load_memory_variables({}), "/n")

print("--- Example B: Conversation Chain with Buffer Memory ---")

conversation = ConversationChain(llm=llm, memory=ConversationBufferMemory())

print("Starting conversation. (Notice how it remembers the first prompt in the second!)")

response1 = conversation.invoke(input="I have a fine chestnut horse named Thunder.")
print(f"Turn 1 (Human) : I have a fine chestnut horse named Thunder.")
print(f"Turn 1 (AI) : {response1}")

response2 = conversation.predict(input="What was the name of my horse again?")
print(f"Turn 2 (Human): What was the name of my horse again?")
print(f"Turn 2 (AI): {response2}")
