#!/usr/bin/python3
# --*-- coding: utf-8 --*--
# @Author: gitsilence
# @Time: 2025/7/2 16:30

import asyncio
from autogen_core import CancellationToken
from autogen_agentchat.agents import UserProxyAgent
from autogen_agentchat.messages import TextMessage
from autogen_agentchat.ui import Console


async def run():
    agent = UserProxyAgent("user_proxy")
    await Console(agent.run_stream(task="你是谁？"))
    # response = await asyncio.create_task(
    #     agent.on_messages(
    #         [TextMessage(content="你的名字什么？", source="user")],
    #         cancellation_token=CancellationToken()
    #     )
    # )
    # print(f"你的名字是：{response.chat_message.content}")
    pass


if __name__ == '__main__':
    asyncio.run(run())
    pass
