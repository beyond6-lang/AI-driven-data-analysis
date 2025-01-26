
from phi.agent import Agent
from phi.model.openai import OpenAIChat
from phi.storage.agent.sqlite import SqlAgentStorage
from phi.tools.duckduckgo import DuckDuckGo
from phi.tools.newspaper4k import Newspaper4k
from phi.run.response import RunEvent, RunResponse
from phi.model.groq import Groq
import os
import sqlalchemy
from sqlalchemy.exc import OperationalError


# Set API key for LLM
os.environ["GROQ_API_KEY"] = "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"  # Replace with your actual Llama 3 API key

# Initialize LLM for PandasAI and Llama
llama_llm = Groq(id="llama-3.3-70b-versatile"),

# Check and handle existing tables
def ensure_table_exists(storage):
    try:
        storage.table.create(storage.db_engine, checkfirst=True)
        print("Table checked and created if not exists.")
    except OperationalError as e:
        print(f"OperationalError occurred: {e}. Table might already exist or there is an issue.")
        # Optionally re-raise if you want to exit on this error
        raise

# Initialize SqlAgentStorage
agent_storage = SqlAgentStorage(table_name="agent_sessions", db_file="tmp/agent.db")

# Ensure table exists before creating the agent
ensure_table_exists(agent_storage)

# Initialize the agent
agent = Agent(
    model=Groq(id="llama-3.3-70b-versatile"),
    # tools=[DuckDuckGo(), Newspaper4k()],
    tools=[],  
    description="You are a researcher writing an article on a topic.",
    instructions=[
        "For a given topic, search for the top 5 links.",
        "Then read each URL and extract the article text.",
        "Analyse and prepare 3-5 bullet points based on the information.",
    ],
    markdown=True,
    show_tool_calls=True,
    add_datetime_to_instructions=True,
    add_history_to_messages=True,
    storage=agent_storage,
)

# As a streaming response generator
def as_stream(response):
    for chunk in response:
        if isinstance(chunk, RunResponse) and isinstance(chunk.content, str):
            if chunk.event == RunEvent.run_response:
                yield chunk.content



# from phi.agent import Agent
# from phi.model.openai import OpenAIChat
# from phi.storage.agent.sqlite import SqlAgentStorage
# from phi.tools.duckduckgo import DuckDuckGo
# from phi.tools.newspaper4k import Newspaper4k
# from phi.run.response import RunEvent, RunResponse
# from phi.model.groq import Groq
# from phi.agent import Agent, RunResponse
# import os



# # Set API key for LLM
# os.environ["GROQ_API_KEY"] = "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" # Replace with your actual Llama 3 API key
 
# # Initialize LLM for PandasAI and Llama
# llama_llm = Groq(id="llama-3.3-70b-versatile"),

# agent = Agent(
#   #model=OpenAIChat(model="gpt-4o"),
#   model= Groq(id="llama-3.3-70b-versatile"),
#   tools=[DuckDuckGo(), Newspaper4k()],
#   description="You are a researcher writing an article on a topic.",
#   instructions=[
#     "For a given topic, search for the top 5 links.",
#     "Then read each URL and extract the article text.",
#     "Analyse and prepare 3-5 bullet points based on the information.",
#   ],
#   markdown=True, show_tool_calls=True,
#   add_datetime_to_instructions=True, add_history_to_messages=True,
#   storage=SqlAgentStorage(table_name="agent_sessions", db_file="tmp/agent.db"),
# )

# def as_stream(response):
#   for chunk in response:
#     if isinstance(chunk, RunResponse) and isinstance(chunk.content, str):
#       if chunk.event == RunEvent.run_response:
#         yield chunk.content