from langchain_ollama import ChatOllama
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser

from langchain_core.runnables import RunnableParallel 

llm = ChatOllama(model="llama3")
parser = StrOutputParser()

# print("--- Example A: Simple LCEL Chain ---")
prompt1 = ChatPromptTemplate.from_template("What is the sigil of {house_name}?")
simple_chain = prompt1 | llm | parser
# response1 = simple_chain.invoke({"house_name": "House Targaryen"})
# print(f"Response: {response1}\n")

# print("--- Example B: Sequential Chaining ---")

prompt_1 = ChatPromptTemplate.from_template("Draft a short, one-sentenced knight's oath for a warrior named {knight}")
chain_draft = prompt_1 | llm | parser

prompt_critique = ChatPromptTemplate.from_template("You are a harsh Maester. Critique this knight's oath in one sentence: \n\n{oath}")
seq_chain = {"oath" : chain_draft} | prompt_critique | llm | parser

# res2 = seq_chain.invoke({"knight": "Ser Duncan the Tall"})
# print(f"Response: {res2}\n")

print("--- Example C: Parallel Chaining ---")
strength = ChatPromptTemplate.from_template("What is the greatest strength of {character}? keep it in one sentence.")
weakness = ChatPromptTemplate.from_template("What is the greatest weakness of {character}? keep it in one sentence.")

chain_str = strength | llm | parser
chain_weak = weakness | llm | parser

parallel_chain = RunnableParallel(
    strength=chain_str,
    weakness=chain_weak
)

results3 = parallel_chain.invoke({"character": "Ser Duncan the Tall"})
print(f"Strength: {results3['strength']}")
print(f"Weakness: {results3['weakness']}")
