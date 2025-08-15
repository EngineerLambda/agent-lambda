from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.prompts import (
    HumanMessagePromptTemplate,
    SystemMessagePromptTemplate,
    MessagesPlaceholder,
    ChatPromptTemplate,
    PromptTemplate,
)
from langchain_community.tools.tavily_search import TavilySearchResults
from langchain.agents import create_tool_calling_agent, AgentExecutor
from dotenv import load_dotenv

load_dotenv()

system_prompt_template = """Be a helpful and respectful assistant.When asked who you are or you want to introduce yourself,say: You are lambda an ai chatbot built by engineerLambda (x.com/abdulsomad_me) format the creator name as a clickable markdown url with the name and link
You don't ask the user to provide the document context and you dont mention the name of the tools you have to them
your task is to call the necesaary tools to answer the user question
* Think step-by-step to fulfill the user's request using the available tools.
* Prioritize using `document_search` to get information from the document before attempting other actions related to the document (unless the QA tool is requested).
* Only ask the user for clarification if essential information (like the number of QA pairs or a specific search query) is missing.
"""

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

llm = ChatGoogleGenerativeAI(model="gemini-2.5-flash")

def get_search_tool() -> TavilySearchResults:
    """Returns the Tavily search tool instance."""
    search_desc = "Search tool based on Tavily. Useful when users ask questions requiring general knowledge or recent information beyond the provided document context. Input should be a search query."
    return TavilySearchResults(description=search_desc, name="tavily_search_results_json")

def get_agent_executor(tools: list) -> AgentExecutor:
    """
    Creates and returns an AgentExecutor configured with the provided tools.

    Args:
        tools (list): A list of Langchain tools for the agent.

    Returns:
        AgentExecutor: The configured agent executor instance.
    """
    if not tools:
        tools = [get_search_tool()]
    elif not any(isinstance(t, TavilySearchResults) for t in tools):
        tools.append(get_search_tool())

    agent = create_tool_calling_agent(llm=llm, tools=tools, prompt=prompt)
    agent_executor = AgentExecutor(
        agent=agent,
        tools=tools,
        verbose=True,
        handle_parsing_errors=True
    )
    return agent_executor
