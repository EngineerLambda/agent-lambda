import streamlit as st
from langchain.prompts import PromptTemplate
from langchain_google_genai import GoogleGenerativeAI
from langchain.output_parsers import PydanticOutputParser
from pydantic import BaseModel, Field
from langchain_core.callbacks.base import BaseCallbackHandler
from agent import agent_executor
import time
# --- Database Imports ---
from database import (
    delete_all_sessions_for_user,
    get_chat_sessions,
    create_chat_session,
    prepare_chat_history,
    add_message_to_session,
    update_session_name,
    delete_all_sessions_for_user
)

# --- Placeholder LLM Title Generation ---
class TitleParser(BaseModel):
    title: str = Field(description="Title of the chat session")
    
title_parser = PydanticOutputParser(pydantic_object=TitleParser)


def generate_title_llm(first_message: str) -> str:
    """
    Placeholder function to simulate LLM title generation.
    Replace this with your actual LLM call.
    Example: Ask the LLM to create a short title based on the first_message.
    """
    prompt = PromptTemplate(template="Based on the given message, suggest a suitable title for the chat. Message: {message}. Follow this instruction strictly {format_instructions}", input_variables=["message", "format_instructions"])
    llm = GoogleGenerativeAI(model="gemini-1.5-flash")
    chain = prompt | llm | title_parser
    
    title = chain.invoke({"message": first_message, "format_instructions": title_parser.get_format_instructions()}).title
    
    return title if title else "Chat Session"

# --- Callback Handler (No changes needed here) ---
class StreamlitCallbackHandler(BaseCallbackHandler):
    def __init__(self, status):
        self.status = status
        self.action = None # Initialize action

    def on_agent_action(self, action, **kwargs):
        self.action = action
        tool_input = action.tool_input
        query = tool_input.get("query", "details") if isinstance(tool_input, dict) else tool_input
        self.status.update(label=f"Searching the internet for: `{query}`")

    def on_tool_start(self, serialized, input_str, **kwargs):
         # Safely access tool_input, handle cases where self.action might not be set yet
        query = "details"
        if self.action and isinstance(self.action.tool_input, dict):
            query = self.action.tool_input.get("query", "details")
        elif self.action:
             query = self.action.tool_input # Assuming tool_input is the string itself

        self.status.update(label=f"Processing Search for: `{query}`")


    def on_tool_end(self, output, **kwargs):
        self.status.update(label="Aggregating search results")

# --- Session State Initialization ---
# Use more specific keys and initialize all relevant keys
default_session_state = {
    "logged_in": False,
    "email": "",
    "username": "",
    "current_session_id": None,
    "current_session_title": "",
    "needs_title": False,
    "session_selected": False
}
for key, default_value in default_session_state.items():
    if key not in st.session_state:
        st.session_state[key] = default_value

# --- Authentication ---
if not st.experimental_user.is_logged_in:
    col1, col_main, col3 = st.columns([1, 5, 1])

    with col_main:
        st.header(f"Agent Lambda: AI Assistant app", anchor=False)

        # Create a container for a bordered look
        with st.container(border=True):
            st.markdown("<br>", unsafe_allow_html=True) # Add a little space inside top border
            google_logo_url = "https://upload.wikimedia.org/wikipedia/commons/c/c1/Google_%22G%22_logo.svg"
            st.markdown(
                f"""
                <div style="display: flex; justify-content: center; margin-top: 20px; margin-bottom: 20px;">
                    <img src="{google_logo_url}" width="25" height="25" style="margin-right: 10px; vertical-align: middle;">
                    <!-- The button itself will be rendered below by Streamlit -->
                </div>
                """,
                unsafe_allow_html=True,
            )

            btn_col1, btn_col2, btn_col3 = st.columns([1, 2, 1])
            with btn_col2:
                login_pressed = st.button("Continue with Google", use_container_width=True)

            if login_pressed:
                try:
                    st.login()
                
                except Exception as e:
                    st.error(f"An error occurred during login setup: {e}")

            st.markdown("<br>", unsafe_allow_html=True) # Add space before bottom border

    st.stop() # Stop execution for non-logged-in users after showing the login page
else:
    # Update session state only if not already logged in or user changed
    if not st.session_state.logged_in or st.session_state.email != st.experimental_user.email:
        st.session_state.logged_in = True
        st.session_state.email = st.experimental_user.email
        st.session_state.username = st.experimental_user.name
        # Reset session selection on new login
        st.session_state.current_session_id = None
        st.session_state.current_session_title = ""
        st.session_state.needs_title = False
        st.session_state.session_selected = False


# --- Sidebar ---
with st.sidebar:
    st.header(f"Welcome, {st.session_state.username}")
    
    # new chat
    if st.button("➕ Create New Chat"):
        # Create session with default name
        new_session_id = create_chat_session(user_id=st.session_state.email)
        st.session_state.current_session_id = new_session_id
        st.session_state.current_session_title = "New Chat" # a placeholder name
        st.session_state.needs_title = True 
        st.session_state.session_selected = True
        st.success("New chat created. Send your first message!")
        st.rerun()

    st.divider()

    # list of old chat histories
    sessions_list = list(get_chat_sessions(user_id=st.session_state.email))

    if sessions_list:
        # Create options for selectbox: list of names
        session_options = {name: sid for sid, name in sessions_list} # Map name to id
        session_names = [name for sid, name in sessions_list] # List of names for display

        # Determine the index of the currently selected session
        current_index = 0
        if st.session_state.current_session_id:
            try:
                # Find the name corresponding to the current ID
                current_name = next(name for sid, name in sessions_list if sid == st.session_state.current_session_id)
                current_index = session_names.index(current_name)
            except (StopIteration, ValueError):
                # If current ID/name not found (e.g., deleted), default to first
                current_index = 0
    

        selected_session_name = st.selectbox(
            "Chat History",
            options=session_names,
            index=current_index,
            key="session_selector"
            # on_change=handle_session_change # Optional: use callback for cleaner logic
        )

        # Handle selection change
        selected_session_id = session_options.get(selected_session_name)
        if selected_session_id and selected_session_id != st.session_state.current_session_id:
            st.session_state.current_session_id = selected_session_id
            st.session_state.current_session_title = selected_session_name
            st.session_state.needs_title = False # Existing sessions don't need new titles
            st.session_state.session_selected = True
            st.rerun() # Rerun to load the selected session

    else:
        st.warning("No previous chats found. Create one!")

    st.divider()
    # Logout
    with st.expander("LOGOUT / DELETE ALL SESSIONS"):
        if st.button("Confirm Logout"):
            # Clear relevant session state keys
            # keys_to_clear = [
            #     "logged_in", "email", "username",
            #     "current_session_id", "current_session_title",
            #     "needs_title", "session_selected"
            # ]
            # for key in keys_to_clear:
            #     st.session_state.pop(key, None)
            st.logout()
        if st.button("Delete All Sessions"):
            deleted_count = delete_all_sessions_for_user(st.session_state.email)
            st.success(f"Deleted {deleted_count} session(s) for user.")

            

# --- Main Chat Interface ---
if st.session_state.logged_in and st.session_state.session_selected and st.session_state.current_session_id:
    st.subheader(f"Chat: {st.session_state.current_session_title}") # Display current session title

    # Load chat history using session_id
    messages = prepare_chat_history(
        session_id=st.session_state.current_session_id,
        chat_history_limit=50 
    )

    # Display chat history
    for message in messages:
        with st.chat_message(message.type):
             st.markdown(message.content)

    # Input to agent
    prompt = st.chat_input("Ask your questions...")
    if prompt:
        # Display user message immediately
        st.chat_message("human").markdown(prompt)

        # --- Title Generation Logic ---
        session_id_to_use = st.session_state.current_session_id
        if st.session_state.needs_title:
            try:
                with st.spinner("Generating title..."):
                    suggested_title = generate_title_llm(prompt) # Call your LLM function
                update_session_name(session_id_to_use, suggested_title)
                st.session_state.current_session_title = suggested_title
                st.session_state.needs_title = False
                st.success(f"Chat title set to: '{suggested_title}'")
                
                time.sleep(1.5)
                
            except Exception as title_e:
                st.error(f"Failed to generate title: {title_e}")
                st.session_state.needs_title = False # Avoid retrying on next message


        # Add user message to DB *before* calling agent
        add_message_to_session(
            session_id=session_id_to_use,
            content=prompt,
            kind="user"
        )
        # Reload history *after* adding user message so agent sees it
        messages = prepare_chat_history(
            session_id=session_id_to_use,
            chat_history_limit=50
        )

        # --- Agent Invocation ---
        with st.status("Processing your request...", expanded=True) as status:
            try:
                callback_handler = StreamlitCallbackHandler(status)
                response = agent_executor.invoke(
                    {
                        "input": prompt,
                        "chat_history": messages # Pass updated history
                    },
                    config={"callbacks": [callback_handler]} # Removed return_intermediate_steps unless needed
                )

                output = response.get("output", "Sorry, I couldn't process that.") # Safely get output
                status.update(label="Done!", state="complete", expanded=True)

                # Display AI response
                with st.chat_message("ai"):
                    st.markdown(output)

                # Add AI response to DB
                add_message_to_session(
                    session_id=session_id_to_use,
                    content=output,
                    kind="ai"
                )

                # If the title was just generated, rerun now to update UI fully
                if not st.session_state.needs_title and session_id_to_use == st.session_state.current_session_id and 'suggested_title' in locals():
                     st.rerun()


            except Exception as e:
                error_message = f"An error occurred: {e}. Please try again later."
                # Check for specific rate limit errors if possible
                if "rate limit" in str(e).lower():
                     error_message = "I am sorry, I seem to have hit a rate limit. Can you try again later?"

                status.update(label=f"Error: {e}", state="error", expanded=True)
                st.chat_message("ai").error(error_message) # Use st.error for visibility

                # Add error message to DB as AI response
                add_message_to_session(
                    session_id=session_id_to_use,
                    content=error_message,
                    kind="ai"
                )
                # If the title was just generated, rerun now to update UI fully
                if not st.session_state.needs_title and session_id_to_use == st.session_state.current_session_id and 'suggested_title' in locals():
                     st.rerun()


elif st.session_state.logged_in and not st.session_state.session_selected:
    st.info("Please create a new chat or select a previous one from the sidebar.")

