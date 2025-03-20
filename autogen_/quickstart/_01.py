#!/usr/bin/python3
# --*-- coding: utf-8 --*--
# @Author: gitsilence
# @Time: 2025/3/20 13:30
import asyncio
from autogen_agentchat.agents import AssistantAgent
from autogen_ext.models.openai import OpenAIChatCompletionClient
import base
import os

async def run():
    model_client = OpenAIChatCompletionClient(
        api_key=os.getenv("OPENAI_API_KEY"),
        base_url=os.getenv("OPENAI_BASE_URL"),
        model=os.getenv("CHAT_MODEL_NAME"),
    )
    agent = AssistantAgent("assistant", model_client=model_client)  # 实例化一个助手角色
    result = await agent.run(task="说 “你好世界！”")
    for message in result.messages:
        print(f"{message.source}: {message.content}")
    await model_client.close()

if __name__ == "__main__":
    asyncio.run(run())
