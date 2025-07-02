#!/usr/bin/python3
# --*-- coding: utf-8 --*--
# @Author: gitsilence
# @Time: 2025/5/19 08:39
"""
RoundRobinGroupChat 参与者轮流回答用户的问题，一轮回答结束之后还会继续循环
除非我们设置了终止条件，或者最大轮次（max_turns）。
"""
import asyncio
from base.base_llm import autogen_llm
from autogen_agentchat.agents import AssistantAgent
from autogen_agentchat.teams import RoundRobinGroupChat
from autogen_agentchat.conditions import TextMentionTermination
from autogen_agentchat.ui import Console
from autogen_ext.agents.web_surfer import MultimodalWebSurfer


async def run():
    planner_agent = AssistantAgent(
        "planner_agent",
        model_client=autogen_llm,
        description="A helpful assistant that can plan trips.",
        system_message="You are a helpful assistant that can suggest a travel plan for a user based on their request.",
    )

    local_agent = AssistantAgent(
        "local_agent",
        model_client=autogen_llm,
        description="A local assistant that can suggest local activities or places to visit.",
        system_message="You are a helpful assistant that can suggest authentic and interesting local activities or places to visit for a user and can utilize any context information provided.",
    )

    language_agent = AssistantAgent(
        "language_agent",
        model_client=autogen_llm,
        description="A helpful assistant that can provide language tips for a given destination.",
        system_message="You are a helpful assistant that can review travel plans, providing feedback on important/critical tips about how best to address language or communication challenges for the given destination. If the plan already includes language tips, you can mention that the plan is satisfactory, with rationale.",
    )

    travel_summary_agent = AssistantAgent(
        "travel_summary_agent",
        model_client=autogen_llm,
        description="A helpful assistant that can summarize the travel plan.",
        system_message="You are a helpful assistant that can take in all of the suggestions and advice from the other agents and provide a detailed final travel plan. You must ensure that the final plan is integrated and complete. YOUR FINAL RESPONSE MUST BE THE COMPLETE PLAN. When the plan is complete and all perspectives are integrated, you can respond with TERMINATE.",
    )
    termination = TextMentionTermination("TERMINATE")
    # termination = TextMentionTermination("ok")

    # RoundRobinGroupChat，Agent轮流发言
    team = RoundRobinGroupChat([planner_agent, local_agent, language_agent, travel_summary_agent],
                               termination_condition=termination)  # 这里max_turns=3了话，就会导致最后一个Agent无法发言
    await Console(team.run_stream(task="准备去杭州旅游3天"))
    await autogen_llm.close()
    # result = await team.run(task="查下当前新浪微博热搜榜前10")
    # for message in result.messages:
    #     print(f"{message.source}: {message.content}")
    pass

if __name__ == '__main__':
    asyncio.run(run())
