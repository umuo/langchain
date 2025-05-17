#!/usr/bin/python3
# --*-- coding: utf-8 --*--
# @Author: gitsilence
# @Time: 2025/3/20 13:46
import asyncio
from autogen_agentchat.agents import UserProxyAgent
from autogen_agentchat.conditions import TextMentionTermination
from autogen_agentchat.teams import RoundRobinGroupChat
from autogen_agentchat.ui import Console
from autogen_ext.agents.web_surfer import MultimodalWebSurfer
from base.base_llm import autogen_llm
"""
1、打开一个浏览器
2、访问bing.com
3、搜索用户的问题
4、获取搜索结果
5、用户输入 exit 结束整个进程
"""


async def main():
    # 创建一个网页浏览代理，设置无头模式为False(显示浏览器界面)，启用动作动画
    web_surfer = MultimodalWebSurfer("web_surfer", headless=False, animate_actions=True,
                                     model_client=autogen_llm)

    # 创建一个用户代理，用于代表用户与系统交互
    user_proxy = UserProxyAgent("user_proxy")

    # 设置终止条件，当用户代理发送"exit"时终止对话
    termination = TextMentionTermination("exit", sources=["user_proxy"])

    # 创建轮询组聊天，包含网页浏览代理和用户代理
    team = RoundRobinGroupChat([web_surfer, user_proxy], termination_condition=termination)

    try:
        # 启动控制台界面，运行任务流：收集当天新闻并总结
        await Console(team.run_stream(task="收集当天的新闻，并总结，标注好新闻的时间"))

    finally:
        # 确保关闭网页浏览代理和LLM连接
        await web_surfer.close()
        await autogen_llm.close()

if __name__ == "__main__":
    # 异步运行主函数
    asyncio.run(main())
