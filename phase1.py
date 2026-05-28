from langchain_ollama import OllamaLLM

llm = OllamaLLM(model="llama3")

prompt = "In one short sentence, who was ser Duncan the tall?"
print("COnsulting the Maester...\n")
response = llm.invoke(prompt)
print(response)