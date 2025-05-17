#!/usr/bin/python3
# --*-- coding: utf-8 --*--
# @Author: gitsilence
# @Time: 2025/3/20 13:30
import asyncio
from autogen_agentchat.agents import AssistantAgent
from base.base_llm import autogen_llm


async def run():
    agent = AssistantAgent("assistant", model_client=autogen_llm)  # 实例化一个助手角色
    result = await agent.run(task="说 “你好世界！”")
    for message in result.messages:
        print(f"{message.source}: {message.content}")
    await autogen_llm.close()

if __name__ == "__main__":
    asyncio.run(run())
