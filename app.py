import streamlit as st
from langchain_core.messages import AIMessage, HumanMessage
from langchain_core.callbacks.base import BaseCallbackHandler
from agent import agent_executor
from database import get_chat_sessions, create_chat_session, prepare_chat_history, add_message_to_session
import asyncio

# Custom Streamlit callback to surface tool usage and agent events
class StreamlitCallbackHandler(BaseCallbackHandler):
    def __init__(self, status):
        self.status = status

    def on_agent_action(self, action, **kwargs):
        self.action = action
        self.status.text(f"Searching the internet for: `{action.tool_input.get("query")}`")

    def on_tool_start(self, tool, input_str, **kwargs):
        self.status.text(f"Processing Search for: `{self.action.tool_input.get("query")}`")

    def on_tool_end(self, output, **kwargs):
        self.status.text("Aggregating search results")

# Manage session state
if "logged_in" not in st.session_state:
    st.session_state.logged_in = False
    st.session_state.email = ""
    st.session_state.username = ""

if "session_selected" not in st.session_state:
    st.session_state.session_selected = False
    st.session_state.session_title = ""

# --- AUTHENTICATION ---
if not st.experimental_user.is_logged_in:
    st.title("Agent Lambda")
    st.info("Please log in to continue. Without login, chats are not saved.")

    if st.button("Continue with Google"):
        st.login("google")
    st.stop()
else:
    st.session_state.logged_in = True
    st.session_state.email = st.experimental_user.email
    st.session_state.username = st.experimental_user.name

# --- SIDEBAR ---
with st.sidebar:
    st.header(f"Welcome, {st.session_state.username}")

    # Create new chat session
    with st.expander("Create New Chat"):
        with st.form("session_create"):
            session_title = st.text_input("New session title")
            submit_new_session = st.form_submit_button("Create New Session")
            if submit_new_session:
                if session_title:
                    create_chat_session(user_id=st.session_state.email, session_name=session_title)
                    st.session_state.session_title = session_title
                    st.session_state.session_selected = True
                    st.success(f"New session '{session_title}' created and selected.")
                else:
                    st.error("Session title cannot be empty.")

    # Select existing sessions
    sessions = list(get_chat_sessions(user_id=st.session_state.email))[::-1]
    if sessions:
        selected_session = st.selectbox(
            "Click to change to other previous chats",
            options=sessions,
            index=sessions.index(st.session_state.session_title) if st.session_state.session_title in sessions else 0,
            key="session_selector"
        )
        if selected_session and selected_session != st.session_state.session_title:
            st.session_state.session_title = selected_session
            st.session_state.session_selected = True
    else:
        st.warning("No sessions found. Create a new one to get started.")

    # Logout section
    with st.expander("LOGOUT"):
        st.warning("Are you sure you want to logout?")
        if st.button("Confirm Logout"):
            for key in ["logged_in", "email", "username", "session_selected", "session_title"]:
                st.session_state.pop(key, None)
            st.logout()
            st.rerun()

# --- MAIN CHAT INTERFACE ---
if st.session_state.logged_in and st.session_state.session_selected:
    st.title("Agent Lambda")

    # Retrieve chat history
    messages = prepare_chat_history(
        user_id=st.session_state.email,
        session_name=st.session_state.session_title,
        chat_history_limit=50
    )
    for message in messages:
        if isinstance(message, HumanMessage):
            st.chat_message("human").markdown(message.content)
        elif isinstance(message, AIMessage):
            st.chat_message("assistant").markdown(message.content)

    # Chat input
    prompt = st.chat_input("Ask your questions ...")
    if prompt:
        st.chat_message("user").markdown(prompt)
        add_message_to_session(
            user_id=st.session_state.email,
            session_name=st.session_state.session_title,
            content=prompt,
            kind="user"
        )

        with st.status("Processing your request...", expanded=True) as status:
            try:
                status.text("Analysing your query...")
                callback_handler = StreamlitCallbackHandler(status)

                response = agent_executor.invoke(
                    {
                        "input": prompt,
                        "chat_history": messages
                    },
                    config={"callbacks": [callback_handler], "return_intermediate_steps": True}
                )

                output = response["output"]
                status.text("Done! Displaying response...")

                with st.chat_message("ai"):
                    st.markdown(output)

                add_message_to_session(
                    user_id=st.session_state.email,
                    session_name=st.session_state.session_title,
                    content=output,
                    kind="ai"
                )

            except Exception as e:
                error_message = "I am sorry, I am currently unable to help with that. Can you rephrase?"
                status.text(f"An error occurred: {e}")
                st.chat_message("ai").write(error_message)
                add_message_to_session(
                    user_id=st.session_state.email,
                    session_name=st.session_state.session_title,
                    content=error_message,
                    kind="ai"
                )
