import streamlit as st
# FIX: Correct import path
from langchain_core.messages import HumanMessage, AIMessage 
from chat import model

# Configuration for LangGraph persistence
CONFIG = {'configurable': {'thread_id': 'thread-1'}}

st.title("LangGraph Chatbot")

# 1. Fetch current state directly from LangGraph to render history
# This removes the need to manually track state in st.session_state['message_history']
current_state = model.get_state(CONFIG)
chat_history = current_state.values.get("messages", []) if current_state.values else []

# Render existing history cleanly
for msg in chat_history:
    if isinstance(msg, HumanMessage):
        with st.chat_message("user"):
            st.markdown(msg.content)
    elif isinstance(msg, AIMessage):
        with st.chat_message("assistant"):
            st.markdown(msg.content)

# 2. Handle new user input
user_input = st.chat_input("Ask me anything!")
if user_input:
    # Display the user message right away
    with st.chat_message("user"):
        st.markdown(user_input)
    
    # Display assistant placeholder and stream the response
    with st.chat_message("assistant"):
        # FIX: Wrapped the HumanMessage inside the proper state dictionary format
        input_state = {"messages": [HumanMessage(content=user_input)]}
        
        # Stream chunks to the UI natively
        ai_message = st.write_stream(
            message_chunk.content 
            for message_chunk, metadata in model.stream(
                input_state, 
                config=CONFIG, 
                stream_mode="messages"
            )
        )