#!/usr/bin/python3
# --*-- coding: utf-8 --*--
# @Author: gitsilence
# @Time: 2025/3/20 13:46
import base
import os
import asyncio
from autogen_agentchat.agents import UserProxyAgent
from autogen_agentchat.conditions import TextMentionTermination
from autogen_agentchat.teams import RoundRobinGroupChat
from autogen_agentchat.ui import Console
from autogen_ext.models.openai import OpenAIChatCompletionClient
from autogen_ext.agents.web_surfer import MultimodalWebSurfer

async def main():
    model_client = OpenAIChatCompletionClient(
        api_key=os.getenv("OPENAI_API_KEY"),
        base_url=os.getenv("OPENAI_BASE_URL"),
        model=os.getenv("CHAT_MODEL_NAME"),
    )

    web_surfer = MultimodalWebSurfer("web_surfer", headless=False, animate_actions=True,
                                     model_client=model_client)

    user_proxy = UserProxyAgent("user_proxy")

    termination = TextMentionTermination("exit", sources=["user_proxy"])

    team = RoundRobinGroupChat([web_surfer, user_proxy], termination_condition=termination)

    try:
        await Console(team.run_stream(task="收集当天的新闻，并总结，标注好新闻的时间"))

    finally:
        await web_surfer.close()
        await model_client.close()

if __name__ == "__main__":
    asyncio.run(main())