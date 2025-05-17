#!/usr/bin/python3
# --*-- coding: utf-8 --*--
# @Author: gitsilence
# @Time: 2025/5/17 12:23

"""
1、获取热点信息（调用MCP获取）
2、热点信息总结
3、请求api，将总结后的信息通过api发送qq群中
"""
import asyncio
from base.base_llm import autogen_llm
from autogen_agentchat.agents import AssistantAgent
from autogen_core.models import UserMessage


async def main():
    chat_agent = AssistantAgent(
        name="chat_assistant",
        model_client=autogen_llm,
        system_message="你是一位乐于助人的助手。",
        # tools=
    )
    # resp = await chat_agent.run(task="你是谁？")  # 通过Agent方式调用模型
    resp = await autogen_llm.create([UserMessage(content="你是谁？", source="user")])
    print(resp)
    await autogen_llm.close()
    pass

if __name__ == '__main__':
    asyncio.run(main())
    pass
