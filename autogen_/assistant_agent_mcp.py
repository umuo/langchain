#!/usr/bin/python3
# --*-- coding: utf-8 --*--
# @Author: gitsilence
# @Time: 2025/5/17 14:06
"""
MCP Workbench
"""
import asyncio
from base.base_llm import autogen_llm
from autogen_agentchat.agents import AssistantAgent
from autogen_ext.tools.mcp import McpWorkbench, StdioServerParams

hot_topic_mcp = StdioServerParams(
    command="/Users/gitsilence/miniforge3/envs/machine_learning/bin/python",
    args=["/Users/gitsilence/PycharmProjects/hot-topic-mcp/main.py"]
)


async def run():
    async with McpWorkbench(hot_topic_mcp) as workbench:
        assistant_agent = AssistantAgent("assistant", model_client=autogen_llm, workbench=workbench)
        result = await assistant_agent.run(task="查一下当前新浪微博的热搜，取前5条")
        for message in result.messages:
            print(f"{message.source}: {message.content}")
        pass
    await autogen_llm.close()


if __name__ == '__main__':
    asyncio.run(run())
    pass
