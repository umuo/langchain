#!/usr/bin/python3
# --*-- coding: utf-8 --*--
# @Author: gitsilence
# @Time: 2025/4/16 10:27
from langgraph.prebuilt import create_react_agent
from langchain_openai import ChatOpenAI
from tools_list import get_zhihu_hot_topics, get_sina_hot_topics
from dotenv import load_dotenv
import os
import logging

debug = True

if debug:
    logging.basicConfig(level=logging.DEBUG)

load_dotenv("../../.env.local")

llm = ChatOpenAI(
    model_name=os.getenv("CHAT_MODEL_NAME"),
    base_url=os.getenv("OPENAI_BASE_URL"),
    api_key=os.getenv("OPENAI_API_KEY"),
    temperature=0, verbose=debug
)
# resp = llm.invoke("你好")
# print(resp)
agent = create_react_agent(llm, [get_zhihu_hot_topics, get_sina_hot_topics], debug=debug)

if __name__ == '__main__':

    while True:
        question = input("请输入问题: ")
        for chunk in agent.stream({"messages": [("user", question)]}, stream_mode="values"):
            message = chunk["messages"][-1]
            if isinstance(message, tuple):
                print(message)
            else:
                message.pretty_print()
