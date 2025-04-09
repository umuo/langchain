#!/usr/bin/python3
# --*-- coding: utf-8 --*--
# @Author: gitsilence
# @Time: 2025/4/9 00:45
import os
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from langchain.agents import create_tool_calling_agent, AgentExecutor
from tools_01 import calculator_tool_02
from langchain_core.prompts import ChatPromptTemplate


load_dotenv('../../.env.local')
API_KEY = os.getenv("OPENAI_API_KEY")
BASE_URL = os.getenv("OPENAI_BASE_URL")
CHAT_MODEL = os.getenv("CHAT_MODEL_NAME")

llm = ChatOpenAI(
    model=CHAT_MODEL,
    api_key=API_KEY,
    base_url=BASE_URL
)
tools = [calculator_tool_02]
prompt = ChatPromptTemplate.from_messages([
    ("system", "你是一个数学助手"),
    ("user", "{input}"),
    ("placeholder", "{agent_scratchpad}"),
])
agent = create_tool_calling_agent(llm, tools, prompt)
agent_executor = AgentExecutor(agent=agent, tools=tools, verbose=True)
response = agent_executor.invoke({"input": "(9 * 9 - 2 * 2) / 7的结果是多少？"})
print(f"""
input: {response["input"]}
output: {response["output"]}
""")

