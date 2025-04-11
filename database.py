# from functools import lru_cache
import streamlit as st
from pymongo import MongoClient
from langchain_core.messages import AIMessage, HumanMessage
from bson.objectid import ObjectId
import datetime
import os

@st.cache_resource
def prepare_db_coll(coll_name):
    # Ensure DATABASE_URL is set in your environment variables
    db_url = os.getenv("DATABASE_URL")
    if not db_url:
        st.error("DATABASE_URL environment variable not set.")
        st.stop()
    client = MongoClient(db_url)
    # Consider adding error handling for connection issues
    try:
        # The ismaster command is cheap and does not require auth.
        client.admin.command('ismaster')
    except:
        st.error("MongoDB server not available.")
        st.stop()

    database = client["lambda"] # Consider making db name configurable
    collection = database[coll_name]
    return collection

message_collection = prepare_db_coll("chat_history")
feedback_collection = prepare_db_coll("feedbacks")

# --- Session Management ---

def create_chat_session(user_id: str, session_name: str = "New Chat"):
    """Creates a new chat session with a default or provided name."""
    session_id = ObjectId()
    timestamp = datetime.datetime.now(datetime.timezone.utc) # Use timezone-aware datetime
    message_collection.insert_one({
        "_id": session_id, # Use _id for the session's unique identifier
        "user_id": user_id,
        "session_name": session_name,
        "created_at": timestamp,
        "last_updated": timestamp,
        "messages": []
    })
    return session_id

def update_session_name(session_id: ObjectId, new_session_name: str):
    """Updates the name of a specific chat session."""
    message_collection.update_one(
        {"_id": session_id},
        {
            "$set": {
                "session_name": new_session_name,
                "last_updated": datetime.datetime.now(datetime.timezone.utc)
             }
        }
    )

def add_message_to_session(session_id: ObjectId, content: str, kind: str):
    """Adds a message to a chat session identified by its ObjectId."""
    message_collection.update_one(
        {"_id": session_id},
        {
            "$push": {
                "messages": {
                    "content": content,
                    "kind": kind,
                    "timestamp": datetime.datetime.now(datetime.timezone.utc)
                }
            },
            "$set": { # Also update the last_updated timestamp
                 "last_updated": datetime.datetime.now(datetime.timezone.utc)
            }
        }
    )

def get_chat_sessions(user_id: str):
    """Fetches all chat sessions for a user, returning (id, name) tuples."""
    # Sort by creation time, newest first
    sessions = message_collection.find(
        {"user_id": user_id},
        {"_id": 1, "session_name": 1} # Project only needed fields
    ).sort("created_at", -1) # Sort by creation time descending
    # Use yield for memory efficiency if many sessions are expected
    for session in sessions:
        yield (session["_id"], session["session_name"])

def prepare_chat_history(session_id: ObjectId, chat_history_limit: int):
    """Fetches messages from a specific session identified by its ObjectId."""
    session = message_collection.find_one({"_id": session_id})
    chat_history = []

    if session:
        # Get messages, ensuring 'messages' key exists
        messages = session.get("messages", [])
        # Get the latest messages up to the limit
        start_index = max(0, len(messages) - chat_history_limit)
        relevant_messages = messages[start_index:]

        for msg in relevant_messages:
            # Check if 'kind' exists in msg, default to handling potential missing keys
            kind = msg.get("kind")
            content = msg.get("content", "") # Default to empty string if content missing
            if kind == "user":
                chat_history.append(HumanMessage(content=content))
            elif kind == "ai":
                chat_history.append(AIMessage(content=content))

    return chat_history

def delete_all_sessions_for_user(user_id: str):
    """
    Deletes all chat sessions associated with a given user ID.
    """
    result = message_collection.delete_many({"user_id": user_id})
    return result.deleted_count


