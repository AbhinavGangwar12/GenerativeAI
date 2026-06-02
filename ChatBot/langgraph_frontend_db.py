import streamlit as st
from langchain_core.messages import HumanMessage, AIMessage 
from langgraph_backend_db import model, get_all_threads
import uuid 

# --- UTILITY FUNCTIONS ---

def generate_thread_id():
    return str(uuid.uuid4())

def add_thread(thread_id):
    if thread_id not in st.session_state['chat_threads']:
        st.session_state['chat_threads'].append(thread_id)

def reset_chat():
    # 1. Create a brand new thread ID
    new_id = generate_thread_id()
    st.session_state['thread_id'] = new_id
    # 2. Register it so it shows up in our sidebar list
    add_thread(new_id)
    # FIX: Removed the line that was clearing out st.session_state['chat_threads']
    st.rerun()

# --- INITIALIZATION ---

if 'chat_threads' not in st.session_state:
    # st.session_state['chat_threads'] = []
    st.session_state['chat_threads'] = get_all_threads()

if 'thread_id' not in st.session_state:
    st.session_state['thread_id'] = generate_thread_id()

# Make sure our active thread is always registered in the list
add_thread(st.session_state['thread_id'])

# Dynamic config targeting the selected thread
CONFIG = {'configurable': {'thread_id': st.session_state['thread_id']}}

st.title("LangGraph Chatbot")

# Fetch history directly from LangGraph checkpoint for the active thread
current_state = model.get_state(CONFIG)
chat_history = current_state.values.get("messages", []) if current_state.values else []

# --- SIDEBAR UI ---
st.sidebar.title("ChatBot Controls")

if st.sidebar.button("➕ New Chat", use_container_width=True):
    reset_chat()

st.sidebar.markdown("---")
st.sidebar.header("Saved Threads")

# Render past threads dynamically
for idx, thread in enumerate(st.session_state['chat_threads']):
    # Highlight the currently active thread for better UX
    is_active = "🔹 " if thread == st.session_state['thread_id'] else "📁 "
    
    # FIX: Added a unique `key` parameter to prevent widget collisions
    if st.sidebar.button(f"{is_active} Thread {idx+1} ({thread[:8]}...)", key=f"btn_{thread}"):
        st.session_state['thread_id'] = thread
        # FIX: Changed deprecated experimental_rerun to native rerun
        st.rerun()

# --- CHAT AREA UI ---

# Render existing history cleanly
for msg in chat_history:
    if isinstance(msg, HumanMessage):
        with st.chat_message("user"):
            st.markdown(msg.content)
    elif isinstance(msg, AIMessage):
        with st.chat_message("assistant"):
            st.markdown(msg.content)

# Handle new user input
user_input = st.chat_input("Ask me anything!")
if user_input:
    # Display the user message right away
    with st.chat_message("user"):
        st.markdown(user_input)
    
    with st.chat_message("assistant"):
        input_state = {"messages": [HumanMessage(content=user_input)]}
        
        # Stream chunks to the UI natively using the assigned thread configuration
        ai_message = st.write_stream(
            message_chunk.content 
            for message_chunk, metadata in model.stream(
                input_state, 
                config=CONFIG, 
                stream_mode="messages"
            )
        )