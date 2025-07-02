#!/usr/bin/python3
# --*-- coding: utf-8 --*--
# @Author: gitsilence
# @Time: 2025/7/2 15:33
from base.base_llm import autogen_llm
from autogen_agentchat.agents import AssistantAgent
from autogen_agentchat.teams import DiGraphBuilder, GraphFlow
from autogen_agentchat.ui import Console

"""
并行执行 A -> (B, C) -> D
"""


async def run():
    # 创建作者代理
    writer = AssistantAgent("writer", model_client=autogen_llm, system_message="起草一段关于气候变化的简短段落。")

    # 创建两个编辑代理
    editor1 = AssistantAgent("editor1", model_client=autogen_llm, system_message="编辑段落的语法。")

    editor2 = AssistantAgent("editor2", model_client=autogen_llm, system_message="编辑段落的风格。")

    # 创建最终审阅者代理
    final_reviewer = AssistantAgent(
        "final_reviewer",
        model_client=autogen_llm,
        system_message="将语法和风格编辑整合成最终版本。",
    )

    # Build the workflow graph
    builder = DiGraphBuilder()
    builder.add_node(writer).add_node(editor1).add_node(editor2).add_node(final_reviewer)

    # Fan-out from writer to editor1 and editor2
    builder.add_edge(writer, editor1)
    builder.add_edge(writer, editor2)

    # Fan-in both editors into final reviewer
    builder.add_edge(editor1, final_reviewer)
    builder.add_edge(editor2, final_reviewer)

    # Build and validate the graph
    graph = builder.build()

    # Create the flow
    flow = GraphFlow(
        participants=builder.get_participants(),
        graph=graph,
    )

    # Run the workflow
    await Console(flow.run_stream(task="起草一段关于气候变化的简短段落"))


import asyncio
asyncio.run(run())

# ---------- TextMessage (user) ----------
# 起草一段关于气候变化的简短段落
# ---------- TextMessage (writer) ----------
# 气候变化是当前全球面临的最严峻挑战之一，其根本原因在于温室气体的排放不断增加，导致全球气温上升和气候模式的显著变化。冰川融化、海平面上升、极端天气事件频发，以及生态系统的脆弱性加剧，均是气候变化的直接后果。这不仅威胁到自然环境的平衡，还对人类的生活方式、经济发展及社会稳定产生深远影响。为应对这一危机，各国亟需采取切实行动，推行可持续发展的政策，减少碳排放，以保护我们共同的地球家园。
# ---------- TextMessage (editor1) ----------
# 气候变化是当前全球面临的最严峻挑战之一，其根本原因在于温室气体排放的不断增加，导致全球气温上升及气候模式的显著变化。冰川融化、海平面上升、极端天气事件频发，以及生态系统脆弱性加剧，均是气候变化的直接后果。这不仅威胁到自然环境的平衡，还对人类的生活方式、经济发展及社会稳定产生深远影响。为应对这一危机，各国亟需采取切实行动，推行可持续发展政策，减少碳排放，以保护我们共同的地球家园。
# ---------- TextMessage (editor2) ----------
# 气候变化是当今世界面临的最严峻挑战之一，其根源在于温室气体排放的不断上升，导致全球气温升高以及气候模式的剧烈变动。冰川加速融化、海平面不断攀升、极端天气愈发频繁，以及生态系统脆弱性的加剧，都是气候变化所带来的直接后果。这不仅威胁到自然环境的和谐，也深刻影响着人类的生活、经济发展及社会稳定。为应对这一危机，各国迫切需要采取有效行动，实施可持续发展政策，减少碳排放，以保护我们共同的地球家园。
# ---------- TextMessage (final_reviewer) ----------
# 气候变化是当今世界面临的最严峻挑战之一，其根源在于温室气体排放的不断上升，这导致全球气温升高以及气候模式的剧烈变动。冰川加速融化、海平面不断攀升、极端天气事件愈发频繁，以及生态系统脆弱性的加剧，都是气候变化带来的直接后果。这不仅威胁到自然环境的和谐，也深刻影响着人类的生活、经济发展及社会稳定。为应对这一危机，各国迫切需要采取有效行动，实施可持续发展政策，减少碳排放，以保护我们共同的地球家园。
# ---------- StopMessage (DiGraphStopAgent) ----------
# Digraph execution is completed.
