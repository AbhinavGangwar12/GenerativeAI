from langchain_ollama import ChatOllama
from langgraph.graph import StateGraph, START, END
from langgraph.types import interrupt, Command
from typing import TypedDict, Literal, Optional
from pydantic import BaseModel
from langgraph.checkpoint.memory import InMemorySaver

model = ChatOllama(model="mistral")
eval = ChatOllama(model="llama3")

class FeedbackModel(BaseModel):
    feedback: Literal["needs_improvement", "good"]

evaluator = eval.with_structured_output(FeedbackModel)

class State(TypedDict):
    topic: str
    outline: str
    tweet: str
    feedback: Literal["needs_improvement", "good"]
    iteration: int
    max_iterations: int
    approval: str           # Tracks human approval ("yes" or "no")
    human_feedback: str     # Captures specific notes from the human

def generate_outline(state: State) -> dict:
    prompt = f"""You are a social media manager who is creative and funny. Your task is to create an outline for a tweet based on the given {state['topic']}. The outline should be concise and engaging, providing a clear structure for the tweet."""

    response = model.invoke(prompt)
    # Initialize iteration counter here
    return {'outline': response.content, 'iteration': 0}

def generate_tweet(state: State) -> dict:
    # Inject human feedback if it exists so the LLM knows what to fix
    hf = state.get("human_feedback", "")
    hf_prompt = f"\n\nTHE HUMAN REVIEWER REJECTED THE PREVIOUS DRAFT WITH THIS FEEDBACK: {hf}. Please adjust the tweet accordingly." if hf else ""

    prompt = f"""Using the following outline: {state['outline']}, create a concise and engaging tweet about {state['topic']}. The tweet should be creative and funny, capturing the essence of the topic.{hf_prompt}"""

    response = model.invoke(prompt)
    return {'tweet': response.content}

def evaluate_tweet(state: State) -> dict:
    prompt = f"""Evaluate the following tweet: {state['tweet']} based on the topic: {state['topic']}, tone and language. 
    Consider factors such as creativity, engagement, and relevance to the topic."""

    response = evaluator.invoke(prompt)
    
    # Increment iteration count here to prevent infinite loops
    current_iteration = state.get('iteration', 0) + 1
    
    return {'feedback': response.feedback, 'iteration': current_iteration}

def check_iterations(state: State) -> Literal["ask_master", "evaluate_tweet"]:
    max_iterations = state.get('max_iterations', 5)
    current_iteration = state.get('iteration', 0)
    curr_feedback = state.get('feedback', 'needs_improvement')

    # If LLM thinks it's good, or we ran out of retries, send to human
    if current_iteration >= max_iterations or curr_feedback == 'good':
        return "ask_master"
    else:
        return "evaluate_tweet"
    
def ask_master(state: State) -> dict:
    decision = interrupt({
        'type': 'approval',
        'message': f"Do you approve the following tweet: {state['tweet']} for the topic: {state['topic']}?",
        'instructions': "When resuming, provide a dict: {'approval': 'yes'/'no', 'feedback': 'your feedback if no'}"
    })

    # Return the human's response to update the graph's state
    return {
        'approval': decision.get('approval', 'no'),
        'human_feedback': decision.get('feedback', '')
    }

def route_human_approval(state: State) -> Literal["__end__", "generate_tweet"]:
    # Route based on the human's decision stored in state
    if state.get('approval') == 'yes':
        return END
    else:
        return "generate_tweet"
    
checkpointer = InMemorySaver()
graph = StateGraph(State)

graph.add_node("generate_outline", generate_outline)
graph.add_node("generate_tweet", generate_tweet)
graph.add_node("evaluate_tweet", evaluate_tweet)
graph.add_node("ask_master", ask_master)

# --- Define the Edges ---
graph.add_edge(START, "generate_outline")
graph.add_edge("generate_outline", "generate_tweet")

# LLM Evaluation Loop
graph.add_conditional_edges("generate_tweet", check_iterations)
graph.add_edge("evaluate_tweet", "generate_tweet")

# Human-in-the-loop Routing
graph.add_conditional_edges("ask_master", route_human_approval)

app = graph.compile(checkpointer=checkpointer)


# --- Execution Block ---
CONFIG = {'configurable': {'thread_id': 'test-1'}}
initial_state = {
    'topic': "The benefits of using LangGraph for building AI applications"
}

print("Running graph...")
# 1. Run the graph until it hits the interrupt or finishes
app.invoke(initial_state, config=CONFIG)

# 2. Check the current state of the graph to see if it paused
state_snapshot = app.get_state(CONFIG)

# LangGraph stores pending interrupts inside the current tasks
if state_snapshot.tasks and state_snapshot.tasks[0].interrupts:
    # Extract the payload we passed into the interrupt() function
    interrupt_payload = state_snapshot.tasks[0].interrupts[0].value
    
    print("\n" + "="*50)
    print("🛑 HUMAN IN THE LOOP TRIGGERED 🛑")
    print(interrupt_payload['message'])
    print("="*50 + "\n")
    
    # 3. Gather human input
    approval = input("Approve? (yes/no): ").strip().lower()
    feedback = ""
    if approval == 'no':
        feedback = input("Please provide feedback for improvement: ").strip()
    
    # 4. Format the response as a dictionary (matching what ask_master expects)
    human_response = {
        'approval': approval,
        'feedback': feedback
    }
    
    print("\nResuming graph with human feedback...")
    # 5. Resume the graph by passing the dict via Command
    result = app.invoke(Command(resume=human_response), config=CONFIG)
    
    print("\n✅ Final Tweet:\n", result.get('tweet'))

else:
    print("\nGraph finished without interrupting. Final State:", state_snapshot.values)