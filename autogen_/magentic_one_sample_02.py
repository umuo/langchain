#!/usr/bin/python3
# --*-- coding: utf-8 --*--
# @Author: gitsilence
# @Time: 2025/5/19 08:39
"""
magentic one 是一个通用的多Agent智能体系统，用于解决跨各种的开放Web和基于文件的任务。、
Magentic-One 编排器 MagenticOneGroupChat 现在只是一个 AgentChat 团队，
支持所有标准的 AgentChat 代理和功能。同样，Magentic-One 的 MultimodalWebSurfer、
FileSurfer 和 MagenticOneCoderAgent 代理现在可作为 AgentChat 代理广泛使用，
可用于任何 AgentChat 工作流。
"""
import asyncio
from base.base_llm import autogen_llm
from autogen_agentchat.agents import AssistantAgent
from autogen_agentchat.teams import MagenticOneGroupChat
from autogen_agentchat.ui import Console
from autogen_ext.agents.web_surfer import MultimodalWebSurfer


async def run():
    assistant = AssistantAgent(
        name="assistant",
        model_client=autogen_llm,
        system_message="你是一个乐于助人的助手",
    )
    surfer = MultimodalWebSurfer(
        "WebSurfer",
        headless=False,  # 显示打开chrome浏览器，不使用无头模式
        animate_actions=True,  # 展示操作的动作
        model_client=autogen_llm,
    )

    team = MagenticOneGroupChat([assistant, surfer], model_client=autogen_llm, max_turns=3)
    result = await team.run(task="查下当前新浪微博热搜榜前10")
    for message in result.messages:
        print(f"{message.source}: {message.content}")
    pass

if __name__ == '__main__':
    asyncio.run(run())
