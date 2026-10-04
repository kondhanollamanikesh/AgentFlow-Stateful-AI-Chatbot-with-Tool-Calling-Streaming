import streamlit as st
from backend2 import workflow,retrieve_all_threads
from langchain_core.messages import HumanMessage,AIMessage,ToolMessage
import time
import uuid


def thread_id_generation():
    thread_id=uuid.uuid4()
    return thread_id

def reset_chat():
    thread_id=thread_id_generation()
    st.session_state['thread_id']=thread_id
    add_thread_id(st.session_state['thread_id'])
    st.session_state['message_history']=[]


if 'message_history' not in st.session_state:
    st.session_state['message_history']=[]

if 'thread_id' not in st.session_state:
    st.session_state['thread_id']=thread_id_generation()

if 'chat_thread' not in  st.session_state:
    st.session_state['chat_thread']=retrieve_all_threads()

def add_thread_id(thread_id):
    if thread_id not in st.session_state['chat_thread']:
        st.session_state['chat_thread'].append(thread_id)

def load_convo(thread_id):

    state = workflow.get_state(
        config={
            'configurable': {
                'thread_id': thread_id
            }
        }
    )

    return state.values.get('messages', [])

st.sidebar.title('**LANGGRAPH CHATBOT**')
if st.sidebar.button('NEW CHAT'):
    reset_chat()


st.sidebar.header("MY PAST CONVERSATIONS")

for thread_id in st.session_state['chat_thread'][::-1]:

    # Load conversation
    messages = load_convo(thread_id)

    # Find first human message
    first_human_message = None

    for message in messages:
        if isinstance(message, HumanMessage):
            first_human_message = message.content
            break

    # Use first human message as sidebar button text
    if first_human_message:

        # Optional: limit the length
        button_name = first_human_message[:30]

        if len(first_human_message) > 30:
            button_name += "..."

    else:
        button_name = "New Conversation"

    # Button
    if st.sidebar.button(button_name, key=thread_id):

        # IMPORTANT: still store the actual thread_id
        st.session_state['thread_id'] = thread_id

        temp_message = []

        for message in messages:

            if isinstance(message, HumanMessage):
                role = 'user'
            else:
                role = 'Assistant'

            temp_message.append({
                'role': role,
                'content': message.content
            })

        st.session_state['message_history'] = temp_message

        st.rerun()

for message in st.session_state['message_history']:
    with st.chat_message(message['role']):
        st.text(message['content'])


user_input= st.chat_input('Ask Anything You Want')

# config={'configurable':{'thread_id':st.session_state['thread_id']}}


config = {
        "configurable": {"thread_id": st.session_state["thread_id"]},
        "metadata": {
            "thread_id": st.session_state["thread_id"]
        },
        "run_name": "chat_turn",
    }

if user_input:
    st.session_state['message_history'].append({'role':'user','content':user_input})
    with st.chat_message('user'):
        st.text(user_input)
        time.sleep(1)
    
    with st.chat_message("assistant"):
        # Use a mutable holder so the generator can set/modify it
        status_holder = {"box": None}

        def ai_only_stream():
            for message_chunk, metadata in workflow.stream(
                {"messages": [HumanMessage(content=user_input)]},
                config=config,
                stream_mode="messages",
            ):
                # Lazily create & update the SAME status container when any tool runs
                if isinstance(message_chunk, ToolMessage):
                    tool_name = getattr(message_chunk, "name", "tool")
                    if status_holder["box"] is None:
                        status_holder["box"] = st.status(
                            f"🔧 Using `{tool_name}` …", expanded=True
                        )
                    else:
                        status_holder["box"].update(
                            label=f"🔧 Using `{tool_name}` …",
                            state="running",
                            expanded=True,
                        )

                # Stream ONLY assistant tokens
                if isinstance(message_chunk, AIMessage):
                    yield message_chunk.content

        ai_message = st.write_stream(ai_only_stream())

        # Finalize only if a tool was actually used
        if status_holder["box"] is not None:
            status_holder["box"].update(
                label="✅ Tool finished", state="complete", expanded=False
            )

    # Save assistant message
    st.session_state["message_history"].append(
        {"role": "assistant", "content": ai_message}
    )