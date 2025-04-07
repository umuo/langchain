#!/usr/bin/python3
# --*-- coding: utf-8 --*--
# @Author: gitsilence
# @Time: 2025/4/7 09:59
import os
from dotenv import load_dotenv
# --- Dependencies ---
# You need to install these libraries:
# pip install langchain langchain-openai langchain_community tavily-python python-dotenv

# --- Load API Keys ---
# Make sure you have OPENAI_API_KEY and TAVILY_API_KEY in your .env file
# or set them as environment variables.
load_dotenv("../.env.local")

# Check if keys are loaded (optional but recommended)
if not os.getenv("OPENAI_API_KEY"):
    raise ValueError("OPENAI_API_KEY not found in environment variables.")
if not os.getenv("TAVILY_API_KEY"):
    raise ValueError("TAVILY_API_KEY not found in environment variables.")

MODEL_NAME = os.getenv("MODEL_NAME") or "gpt-4o"
MODEL_TOOL_NAME = MODEL_NAME
API_BASE_URL = os.getenv("OPENAI_BASE_URL")
API_KEY = os.getenv("OPENAI_API_KEY")

# --- 1. Initialize the LLM ---
# We'll use OpenAI's GPT model as the "brain" for planning and synthesis.
# You can swap this with other models like Google's Gemini or Anthropic's Claude
# if you have the corresponding integrations and API keys.
from langchain_openai import ChatOpenAI

# Adjust model_name and temperature as needed
# gpt-4-turbo or gpt-4o are generally better at planning
llm = ChatOpenAI(
    model_name=MODEL_NAME,
    base_url=API_BASE_URL,
    api_key=API_KEY,
    temperature=0
)

# --- 2. Define the Tools ---
# The agent needs tools to gather information.
# A web search tool is essential for general research.
# Tavily is a search engine optimized for LLM agents.
from langchain_community.tools.tavily_search import TavilySearchResults

# You can add more tools here, for example:
# - A retriever for specific documentation (like LangChain docs)
# - A Python REPL tool for code execution/analysis
# - A database query tool
tools = [
    TavilySearchResults(max_results=3) # Get top 3 results per search query
]


# --- 3. Create the Plan-and-Execute Agent ---
from langchain import hub
# Create the main PlanAndExecute agent
# Note: As of recent LangChain versions, creating PlanAndExecute might be evolving.
# The core idea is combining a planning step with an execution step.
# A simpler approach might use newer agent constructors if PlanAndExecute direct class isn't standard.
# Let's try a direct conceptual implementation:

from langchain.chains import LLMChain
from langchain.schema.runnable import RunnableSequence
from langchain_core.prompts import PromptTemplate

# Define a simple planning prompt (can be much more sophisticated)
planner_prompt_template = """
You are a research assistant. Your goal is to create a step-by-step plan
to answer the user's question comprehensively using available tools.

Available tools: {tool_descriptions}

User Question: {input}

Devise a plan with clear steps. Each step should specify the exact query or action needed for a tool.
Output the plan as a numbered list.

Plan:
"""
# planner_prompt = PromptTemplate.from_template(planner_prompt_template)

# tool_description_string = "\n".join([f"{tool.name}: {tool.description}" for tool in tools])

# planner_chain = LLMChain(llm=llm, prompt=planner_prompt)
# planner_chain = RunnableSequence(planner_prompt | llm)

# This is a conceptual representation. The actual execution flow within
# newer LangChain versions might use different abstractions like LCEL runnables
# or specific agent initializers designed for planning.

# Let's use a more standard agent setup that implicitly handles planning and execution,
# like the OpenAI Tools agent or a ReAct agent, which often achieve a similar outcome.
# The `PlanAndExecute` specific class might be less common now.

# --- Alternative using a capable Agent (like OpenAI Tools Agent) ---
# This often handles multi-step reasoning implicitly better with capable models.
from langchain.agents import AgentExecutor, create_openai_tools_agent

# Ensure the LLM supports tool calling (like gpt-4-turbo, gpt-3.5-turbo with recent versions)
llm_tools = ChatOpenAI(
    model_name=MODEL_TOOL_NAME,
    base_url=API_BASE_URL,
    api_key=API_KEY,
    temperature=0
)

# Get the prompt to use - you can modify this!
# Check LangChain Hub for suitable prompts: https://smith.langchain.com/hub
# Example prompt for agents that use OpenAI tool calling
prompt = hub.pull("hwchase17/openai-tools-agent")
# print(prompt.messages)
formatted_prompt = prompt.format_messages(
    tools=[],  # 可以传入工具列表
    input="示例输入", agent_scratchpad=[]
)
for m in formatted_prompt:
    print(f"role: {m.type} | content: {m.content}")

# Create the agent
agent = create_openai_tools_agent(llm_tools, tools, prompt)

# Create the Agent Executor
agent_executor = AgentExecutor(
    agent=agent,
    tools=tools,
    verbose=True, # Set to True to see the thought process, tool calls, and results
    handle_parsing_errors=True, # Helps prevent crashes if the LLM output isn't perfect
)


# --- 4. Define the Input Question ---
question = "Langchain 如何实现 Deep Research？ 请解释其核心概念、关键组件/工具以及最佳实践。"
# (How does Langchain implement Deep Research? Please explain its core concepts, key components/tools, and best practices.)


# --- 5. Run the Agent Executor ---
print(f"--- Starting Research for: '{question}' ---")

# Use invoke for the standard LangChain interface
try:
    result = agent_executor.invoke({"input": question})
    final_answer = result.get("output", "No output found.") # Get the final answer key
except Exception as e:
    final_answer = f"An error occurred: {e}"

print("\n--- Final Answer ---")
print(final_answer)
