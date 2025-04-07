# from functools import lru_cache
import streamlit as st
from pymongo import MongoClient
from langchain_core.messages import AIMessage, HumanMessage
from bson.objectid import ObjectId
import datetime
import os

@st.cache_resource
def prepare_db_coll(coll_name):
    client = MongoClient(os.getenv("DATABASE_URL"))
    database = client["lambda"]
    collection = database[coll_name]
    return collection

message_collection = prepare_db_coll("chat_history")
# users_collection = prepare_db_coll("user_collection")
feedback_collection = prepare_db_coll("feedbacks")

# more helper functions
def create_chat_session(user_id, session_name):
    session_id = ObjectId()
    message_collection.insert_one({
        "user_id": user_id,
        "session_name": session_name,
        "session_id": session_id,
        "messages": []
    })
    return session_id

# Helper function to add a message to a chat session
def add_message_to_session(user_id, session_name, content, kind):
    message_collection.update_one(
        {
            "user_id": user_id,
            "session_name": session_name
        },
        {
            "$push": {
                "messages": {
                    "content": content,
                    "kind": kind,
                    "timestamp": datetime.datetime.now()
                }
            }
        }
    )


# helper function to fetch all chat sessions for a user
def get_chat_sessions(user_id):
    sessions = list(message_collection.find({"user_id": user_id}))
    for session in sessions:
        yield session["session_name"]

# helper function to fetch messages from a specific session
def prepare_chat_history(user_id: str, session_name: str, chat_history_limit: int):
    # fetching the specific session for the user
    session = message_collection.find_one({"user_id": user_id, "session_name": session_name})
    chat_history = []

    if session:
        messages = session.get("messages", [])
        
        # adding messages to the chat history up to the specified limit
        for msg in messages[:chat_history_limit]:
            if msg["kind"] == "user":
                chat_history.append(HumanMessage(content=msg["content"]))
            elif msg["kind"] == "ai":
                chat_history.append(AIMessage(content=msg["content"]))
    
    return chat_history

