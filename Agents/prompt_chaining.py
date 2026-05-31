from langchain_ollama import ChatOllama
from langgraph.graph import StateGraph, START, END
from typing import TypedDict 
from langchain_core.prompts import ChatPromptTemplate

class LLMState(TypedDict):
    topic: str
    topic_outline: str
    blog: str  

def generate_outline(state: LLMState) -> LLMState:
    llm = ChatOllama(model="mistral", temperature=0.0)
    prompt = ChatPromptTemplate.from_template(
        "Generate a detailed outline for a blog post about {topic}."
    )
    topic = state['topic']
    response = llm.invoke(prompt.invoke({"topic":topic}))
    # state['topic_outline'] = response.content
    return {"topic_outline": response.content}

def generate_blog(state: LLMState) -> LLMState:
    llm = ChatOllama(model="mistral", temperature=0.0)
    prompt = ChatPromptTemplate.from_template(
        "Write a blog post based on the following outline: {topic_outline}"
    )
    topic_outline = state['topic_outline']
    response = llm.invoke(prompt.invoke({"topic_outline":topic_outline}))
    # state['blog'] = response.content
    return {"blog": response.content}

graph = StateGraph(LLMState)

graph.add_node("generate_outline", generate_outline)
graph.add_node("generate_blog", generate_blog)

graph.add_edge(START, "generate_outline")
graph.add_edge("generate_outline", "generate_blog")
graph.add_edge("generate_blog", END)

workflow = graph.compile()

query = {"topic" : "Ser Duncan the Tall in the world of A Song of Ice and Fire"}
output = workflow.invoke(query)
print(output)