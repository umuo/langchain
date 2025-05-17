#!/usr/bin/python3
# --*-- coding: utf-8 --*--
# @Author: gitsilence
# @Time: 2025/5/17 13:52
from base.base_llm import autogen_llm
from autogen_core.tools import FunctionTool
# """A token used to cancel pending async calls"""
from autogen_core import CancellationToken
from autogen_agentchat.agents import AssistantAgent
import asyncio


async def add(a: int, b: int) -> int:
    print(f"函数调用：{a} + {b}")
    return a + b


async def run():
    add_tool = FunctionTool(add, description="求两数之和")
    # 手动调用
    # result = await add_tool.run_json({"a": 1, "b": 2}, CancellationToken())
    assistant_agent = AssistantAgent("assistant", model_client=autogen_llm, tools=[add_tool])
    result = await assistant_agent.run(task="计算 983+331 的结果")
    # print(add_tool.return_value_as_string(result))
    for message in result.messages:
        print(f"{message.source}: {message.content}")
    await autogen_llm.close()
    pass

if __name__ == '__main__':
    asyncio.run(run())
