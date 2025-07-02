#!/usr/bin/python3
# --*-- coding: utf-8 --*--
# @Author: gitsilence
# @Time: 2025/7/2 10:47

import asyncio

from autogen_agentchat.agents import AssistantAgent
from autogen_agentchat.conditions import MaxMessageTermination
from autogen_agentchat.teams import DiGraphBuilder, GraphFlow
from autogen_agentchat.ui import Console
from base.base_llm import autogen_llm

"""
A -> B
A -> C
"""


async def main():
    # 初始化Agent A，检查用户输入是否为中文。如果是，输出“yes”，否则输出“no”。
    agent_a = AssistantAgent(
        "A",
        model_client=autogen_llm,
        system_message="Detect if the input is in Chinese. If it is, say 'yes', else say 'no', and nothing else.",
    )
    # 初始化Agent B，将用户的输入翻译为英文。
    agent_b = AssistantAgent("B", model_client=autogen_llm, system_message="Translate input to English.")
    # 初始化Agent C，将用户的输入翻译为中文。
    agent_c = AssistantAgent("C", model_client=autogen_llm, system_message="Translate input to Chinese.")

    # 构建有向图，条件分支 A -> B ("yes"), A -> C (否则)。
    builder = DiGraphBuilder()
    # 添加节点到图中。
    builder.add_node(agent_a).add_node(agent_b).add_node(agent_c)
    # Create conditions as callables that check the message content.
    # 添加边到图中。从agent_a到agent_b，条件是消息内容包含"yes"
    builder.add_edge(agent_a, agent_b, condition=lambda msg: "yes" in msg.to_model_text())
    # 添加边到图中。从agent_a到agent_c，条件是消息内容不包含"yes"
    # msg.to_model_text() 是一个方法，用于将消息转换为字符串。
    builder.add_edge(agent_a, agent_c, condition=lambda msg: "yes" not in msg.to_model_text())
    # 构建有向图。
    graph = builder.build()

    # Create a GraphFlow team with the directed graph.
    team = GraphFlow(
        # 成员列表
        participants=[agent_a, agent_b, agent_c],
        # 有向图
        # 有向图是一个字典，键是节点，值是一个字典，键是边，值是条件。
        graph=graph,
        # 终止条件，当消息数量达到5时终止。包含用户输入的消息和AI输出的消息。
        termination_condition=MaxMessageTermination(5),
    )

    # Run the team and print the events.

    await Console(team.run_stream(task="AutoGen是一个用于构建AI代理的框架。"))

    # async for event in team.run_stream(task="AutoGen is a framework for building AI agents."):
    #     if not hasattr(event, "messages"):
    #         print(event.source, " -> ", event.content)


asyncio.run(main())
