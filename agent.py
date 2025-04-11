import os
import streamlit as st
from langchain_groq import ChatGroq
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.prompts import PromptTemplate
from langchain_core.prompts import (
    HumanMessagePromptTemplate,
    SystemMessagePromptTemplate,
    MessagesPlaceholder,
    ChatPromptTemplate,
    PromptTemplate,
)
from langchain_community.tools.tavily_search import TavilySearchResults
from langchain.agents import create_tool_calling_agent, AgentExecutor
from dotenv import load_dotenv; load_dotenv()

# System Prompt Template
system_prompt_template = f"""Be a helpful and respectful assitant"""

# Prompt Setup
chat_prompts = [
    SystemMessagePromptTemplate(
        prompt=PromptTemplate(template=system_prompt_template)
    ),
    MessagesPlaceholder(variable_name="chat_history"),
    HumanMessagePromptTemplate(
        prompt=PromptTemplate(template="{input}", input_variables=["input"])
    ),
    MessagesPlaceholder(variable_name="agent_scratchpad"),
]

prompt = ChatPromptTemplate(chat_prompts)
# llm = ChatGroq(model_name="llama-3.3-70b-versatile", temperature=0)
llm = ChatGoogleGenerativeAI(model="gemini-2.0-flash")

search_desc = "search tool based on tavily, useful when users ask questions and you don't have answers to them, such as questions asking for latest info. input should be a search query."
search = TavilySearchResults(description=search_desc)
tools = [search]
agent = create_tool_calling_agent(llm=llm, tools=tools, prompt=prompt)
agent_executor = AgentExecutor(agent=agent, tools=tools, verbose=True)


            