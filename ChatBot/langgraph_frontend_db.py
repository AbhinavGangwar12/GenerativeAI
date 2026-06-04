# import streamlit as st
# from langchain_core.messages import HumanMessage, AIMessage , ToolMessage
# from langgraph_mcp_backend import model, get_all_threads, submit_async_task
# import uuid 
# import dotenv 
# import queue

# dotenv.load_dotenv()  # Load environment variables from .env file, if it exists

# # --- UTILITY FUNCTIONS ---

# def generate_thread_id():
#     return str(uuid.uuid4())

# def add_thread(thread_id):
#     if thread_id not in st.session_state['chat_threads']:
#         st.session_state['chat_threads'].append(thread_id)

# def reset_chat():
#     # 1. Create a brand new thread ID
#     new_id = generate_thread_id()
#     st.session_state['thread_id'] = new_id
#     # 2. Register it so it shows up in our sidebar list
#     add_thread(new_id)
#     # FIX: Removed the line that was clearing out st.session_state['chat_threads']
#     st.rerun()

# # --- INITIALIZATION ---

# if 'chat_threads' not in st.session_state:
#     # st.session_state['chat_threads'] = []
#     st.session_state['chat_threads'] = get_all_threads()

# if 'thread_id' not in st.session_state:
#     st.session_state['thread_id'] = generate_thread_id()

# # Make sure our active thread is always registered in the list
# add_thread(st.session_state['thread_id'])

# # Dynamic config targeting the selected thread
# # CONFIG = {'configurable': {'thread_id': st.session_state['thread_id']}} #this config variable is absolutely fine but to integrate the threads with langsmith, we need to pass the thread_id in the config of the model 
# CONFIG = {
#     'configurable':{'thread_id': st.session_state['thread_id']},
#     'metadata' : {'thread_id': st.session_state['thread_id']}, # Adding thread_id to metadata for better traceability in LangSmith
#     "run_name" : "chat_run " + st.session_state['thread_id'][:8]
# }

# st.title("LangGraph Chatbot")

# # Fetch history directly from LangGraph checkpoint for the active thread
# current_state = model.get_state(CONFIG)
# chat_history = current_state.values.get("messages", []) if current_state.values else []

# # --- SIDEBAR UI ---
# st.sidebar.title("ChatBot Controls")

# if st.sidebar.button("➕ New Chat", use_container_width=True):
#     reset_chat()

# st.sidebar.markdown("---")
# st.sidebar.header("Saved Threads")

# # Render past threads dynamically
# for idx, thread in enumerate(st.session_state['chat_threads']):
#     # Highlight the currently active thread for better UX
#     is_active = "🔹 " if thread == st.session_state['thread_id'] else "📁 "
    
#     # FIX: Added a unique `key` parameter to prevent widget collisions
#     if st.sidebar.button(f"{is_active} Thread {idx+1} ({thread[:8]}...)", key=f"btn_{thread}"):
#         st.session_state['thread_id'] = thread
#         # FIX: Changed deprecated experimental_rerun to native rerun
#         st.rerun()

# # --- CHAT AREA UI ---

# # Render existing history cleanly
# for msg in chat_history:
#     if isinstance(msg, HumanMessage):
#         with st.chat_message("user"):
#             st.markdown(msg.content)
#     elif isinstance(msg, AIMessage):
#         with st.chat_message("assistant"):
#             st.markdown(msg.content)

# # Handle new user input
# user_input = st.chat_input("Ask me anything!")
# # if user_input:
# #     # Display the user message right away
# #     with st.chat_message("user"):
# #         st.markdown(user_input)
    
# #     with st.chat_message("assistant"):
# #         input_state = {"messages": [HumanMessage(content=user_input)]}
        
# #         # Stream chunks to the UI natively using the assigned thread configuration
# #         ai_message = st.write_stream(
# #             message_chunk.content 
# #             for message_chunk, metadata in model.stream(
# #                 input_state, 
# #                 config=CONFIG, 
# #                 stream_mode="messages"
# #             )
# #         )

# if user_input:
#     # Show user's message
#     st.session_state["message_history"].append({"role": "user", "content": user_input})
#     with st.chat_message("user"):
#         st.text(user_input)

#     CONFIG = {
#         "configurable": {"thread_id": st.session_state["thread_id"]},
#         "metadata": {"thread_id": st.session_state["thread_id"]},
#         "run_name": "chat_turn",
#     }

#     # Assistant streaming block
#     with st.chat_message("assistant"):
#         # Use a mutable holder so the generator can set/modify it
#         status_holder = {"box": None}

#         def ai_only_stream():
#             event_queue: queue.Queue = queue.Queue()

#             async def run_stream():
#                 try:
#                     async for message_chunk, metadata in model.astream(
#                         {"messages": [HumanMessage(content=user_input)]},
#                         config=CONFIG,
#                         stream_mode="messages",
#                     ):
#                         event_queue.put((message_chunk, metadata))
#                 except Exception as exc:
#                     event_queue.put(("error", exc))
#                 finally:
#                     event_queue.put(None)

#             submit_async_task(run_stream())

#             while True:
#                 item = event_queue.get()
#                 if item is None:
#                     break
#                 message_chunk, metadata = item
#                 if message_chunk == "error":
#                     raise metadata

#                 # Lazily create & update the SAME status container when any tool runs
#                 if isinstance(message_chunk, ToolMessage):
#                     tool_name = getattr(message_chunk, "name", "tool")
#                     if status_holder["box"] is None:
#                         status_holder["box"] = st.status(
#                             f"🔧 Using `{tool_name}` …", expanded=True
#                         )
#                     else:
#                         status_holder["box"].update(
#                             label=f"🔧 Using `{tool_name}` …",
#                             state="running",
#                             expanded=True,
#                         )

#                 # Stream ONLY assistant tokens
#                 if isinstance(message_chunk, AIMessage):
#                     yield message_chunk.content

#         ai_message = st.write_stream(ai_only_stream())

#         # Finalize only if a tool was actually used
#         if status_holder["box"] is not None:
#             status_holder["box"].update(
#                 label="✅ Tool finished", state="complete", expanded=False
#             )


import streamlit as st
from langchain_core.messages import HumanMessage, AIMessage, ToolMessage
from langgraph_mcp_backend import model, get_all_threads, submit_async_task, run_async
import uuid 
import dotenv 
import queue

dotenv.load_dotenv()

def generate_thread_id():
    return str(uuid.uuid4())

def add_thread(thread_id):
    if thread_id not in st.session_state['chat_threads']:
        st.session_state['chat_threads'].append(thread_id)

def reset_chat():
    new_id = generate_thread_id()
    st.session_state['thread_id'] = new_id
    add_thread(new_id)
    st.rerun()

# --- INITIALIZATION ---
if 'chat_threads' not in st.session_state:
    st.session_state['chat_threads'] = get_all_threads()

if 'thread_id' not in st.session_state:
    st.session_state['thread_id'] = generate_thread_id()

# FIX: Added missing message_history initialization
if 'message_history' not in st.session_state:
    st.session_state['message_history'] = []

add_thread(st.session_state['thread_id'])

CONFIG = {
    'configurable': {'thread_id': st.session_state['thread_id']},
    'metadata': {'thread_id': st.session_state['thread_id']},
    "run_name": "chat_run_" + st.session_state['thread_id'][:8]
}

st.title("LangGraph Chatbot")

# Fetch history via the synchronous loop executor wrapper
current_state = run_async(model.aget_state(CONFIG))
chat_history = current_state.values.get("messages", []) if current_state.values else []

# --- SIDEBAR UI ---
st.sidebar.title("ChatBot Controls")
if st.sidebar.button("➕ New Chat", use_container_width=True):
    reset_chat()

st.sidebar.markdown("---")
st.sidebar.header("Saved Threads")

for idx, thread in enumerate(st.session_state['chat_threads']):
    is_active = "🔹 " if thread == st.session_state['thread_id'] else "📁 "
    if st.sidebar.button(f"{is_active} Thread {idx+1} ({thread[:8]}...)", key=f"btn_{thread}"):
        st.session_state['thread_id'] = thread
        st.rerun()

# --- CHAT AREA UI ---
for msg in chat_history:
    if isinstance(msg, HumanMessage):
        with st.chat_message("user"):
            st.markdown(msg.content)
    elif isinstance(msg, AIMessage) and msg.content: # only render if there is text content
        with st.chat_message("assistant"):
            st.markdown(msg.content)

# Handle new user input
user_input = st.chat_input("Ask me anything!")

if user_input:
    st.session_state["message_history"].append({"role": "user", "content": user_input})
    with st.chat_message("user"):
        st.markdown(user_input) # changed from text to markdown for UI consistency

    with st.chat_message("assistant"):
        status_holder = {"box": None}

        def ai_only_stream():
            event_queue = queue.Queue()

            async def run_stream():
                try:
                    # Target astream on your compiled LangGraph model
                    async for message_chunk, metadata in model.astream(
                        {"messages": [HumanMessage(content=user_input)]},
                        config=CONFIG,
                        stream_mode="messages",
                    ):
                        event_queue.put((message_chunk, metadata))
                except Exception as exc:
                    event_queue.put(("error", exc))
                finally:
                    event_queue.put((None, None))

            submit_async_task(run_stream())

            while True:
                message_chunk, metadata = event_queue.get()
                if message_chunk is None:
                    break
                if message_chunk == "error":
                    raise metadata

                # Dynamically show tools working inside Streamlit UI
                if isinstance(message_chunk, ToolMessage):
                    tool_name = getattr(message_chunk, "name", "tool")
                    if status_holder["box"] is None:
                        status_holder["box"] = st.status(f"🔧 Using `{tool_name}` ...", expanded=True)
                    else:
                        status_holder["box"].update(label=f"🔧 Using `{tool_name}` ...", state="running")

                if isinstance(message_chunk, AIMessage):
                    yield message_chunk.content

        ai_message = st.write_stream(ai_only_stream())

        if status_holder["box"] is not None:
            status_holder["box"].update(label="✅ Tool finished", state="complete", expanded=False)
            
    # Force rerun to sync state properly into standard history on completion
    st.rerun()