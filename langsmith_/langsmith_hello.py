#!/usr/bin/python3
# --*-- coding: utf-8 --*--
# @Author: gitsilence
# @Time: 2025/4/7 13:24
from langchain_openai import ChatOpenAI
from dotenv import load_dotenv
import os

load_dotenv('../.env.local')

llm = ChatOpenAI(
    model_name=os.getenv("CHAT_MODEL_NAME"),
    base_url=os.getenv("OPENAI_BASE_URL"),
    api_key=os.getenv("OPENAI_API_KEY"),
    temperature=0
)
response = llm.invoke("你好，lang smith!")
print(response.content)
