import streamlit as st
from pymongo import MongoClient
from langchain_core.messages import AIMessage, HumanMessage
from bson.objectid import ObjectId
import datetime
import os

@st.cache_resource
def prepare_db_coll(coll_name):
    db_url = os.environ["DATABASE_URL"]
    client = MongoClient(db_url)
    database = client["lambda"]
    collection = database[coll_name]
    return collection

message_collection = prepare_db_coll("chat_history")
feedback_collection = prepare_db_coll("feedbacks")

def create_chat_session(user_id: str, session_name: str = "New Chat"):
    """Creates a new chat session with a unique name derived from the base name and session ID."""
    session_id = ObjectId()
    timestamp = datetime.datetime.now(datetime.timezone.utc)
    base_name = str(session_name) if session_name else "New Chat"
    unique_session_name = f"{base_name}_{session_id}"
    message_collection.insert_one({
        "_id": session_id,
        "user_id": user_id,
        "session_name": unique_session_name,
        "created_at": timestamp,
        "last_updated": timestamp,
        "messages": []
    })
    return session_id, unique_session_name

def update_session_name(session_id: ObjectId, base_session_name: str):
    """Updates the name of a specific chat session, ensuring uniqueness by appending the session ID."""
    base_name = str(base_session_name) if base_session_name else "Chat Session"
    unique_session_name = f"{base_name}_{session_id}"
    result = message_collection.update_one(
        {"_id": session_id},
        {
            "$set": {
                "session_name": unique_session_name,
                "last_updated": datetime.datetime.now(datetime.timezone.utc)
             }
        }
    )
    return unique_session_name if result.modified_count > 0 else None

def add_message_to_session(session_id: ObjectId, content: str, kind: str):
    """Adds a message to a chat session identified by its ObjectId."""
    if not isinstance(session_id, ObjectId):
        try:
            session_id = ObjectId(session_id)
        except Exception:
            return

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
            "$set": {
                 "last_updated": datetime.datetime.now(datetime.timezone.utc)
            }
        }
    )

def get_chat_sessions(user_id: str):
    """Fetches all chat sessions for a user, returning (id, unique_session_name) tuples."""
    sessions = message_collection.find(
        {"user_id": user_id},
        {"_id": 1, "session_name": 1}
    ).sort("created_at", -1)
    for session in sessions:
        unique_name = session.get("session_name", f"Chat_{session['_id']}")
        yield (session["_id"], unique_name)

def prepare_chat_history(session_id: ObjectId, chat_history_limit: int):
    """Fetches messages from a specific session identified by its ObjectId."""
    if not isinstance(session_id, ObjectId):
        try:
            session_id = ObjectId(session_id)
        except Exception:
            return []

    session = message_collection.find_one({"_id": session_id})
    chat_history = []

    if session:
        messages = session.get("messages", [])
        start_index = max(0, len(messages) - chat_history_limit)
        relevant_messages = messages[start_index:]

        for msg in relevant_messages:
            kind = msg.get("kind")
            content = msg.get("content", "")
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

# function to get all unique users and their session counts
def get_unique_users_and_session_counts():
    pipeline = [
        {
            "$group": {
                "_id": "$user_id",
                "count": {"$sum": 1}
            }
        },
        {
            "$sort": {"count": -1}
        }
    ]
    result = message_collection.aggregate(pipeline)
    return [(user["_id"], user["count"]) for user in result]