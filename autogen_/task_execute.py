#!/usr/bin/python3
# --*-- coding: utf-8 --*--
# @Author: gitsilence
# @Time: 2025/7/2 16:42


from autogen_agentchat.agents import CodeExecutorAgent, AssistantAgent
from autogen_ext.code_executors.local import LocalCommandLineCodeExecutor
from base.base_llm import autogen_llm
import os
from jinja2 import Template
import datetime
from autogen_agentchat.teams import MagenticOneGroupChat
from autogen_agentchat.ui import Console
import logging

# logging.basicConfig(level=logging.DEBUG)
#
# logger = logging.getLogger("autogen")
# logger.setLevel(logging.DEBUG)
# file_handler = logging.FileHandler("autogen_debug.log")
#
# formatter = logging.Formatter("[%(asctime)s] %(levelname)s:%(name)s:%(message)s")
# file_handler.setFormatter(formatter)
#
# logger.addHandler(file_handler)
# 你也可以添加控制台输出（可选）
# console_handler = logging.StreamHandler()
# console_handler.setFormatter(formatter)
# logger.addHandler(console_handler)


async def read_prompts(name: str, base_path: str = "/Users/gitsilence/PycharmProjects/ai-study-project/autogen_/prompts") -> str:
    with open(os.path.join(base_path, f"{name}.md"), "r", encoding="utf-8") as f:
        return f.read()


async def get_reporter_agent():
    reporter_prompt = await read_prompts("reporter")
    # 创建 Jinja2 模板对象
    template = Template(reporter_prompt)

    # 获取当前时间
    current_time = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    # 填充模板
    filled_template = template.render(current_time=current_time)

    return AssistantAgent(
        "reporter", description="将内容整理成最终的报告",
        model_client=autogen_llm,
        system_message=filled_template,
    )


async def run():
    code_executor = LocalCommandLineCodeExecutor(work_dir=r'/Users/gitsilence/PycharmProjects/ai-study-project/autogen_/prompts')
    code_agent = CodeExecutorAgent("coder_executor", code_executor=code_executor, model_client=autogen_llm)
    reporter_agent = await get_reporter_agent()

    team = MagenticOneGroupChat(participants=[code_agent, reporter_agent], model_client=autogen_llm)
    await Console(team.run_stream(task="""
任务一步一步执行，任务步骤如下：
1、根据鲁迅的生平，写一篇报告，主要介绍出生、生活背景、小时候的经历、长大后的经历，到去世期间的经历
2、将鲁迅的报告保存为txt文件，通过执行python代码写入到目录 "/Users/gitsilence/PycharmProjects/ai-study-project/autogen_/doc"
3、根据鲁迅的弟弟周作人的升平，写一篇报告，同上
4、将周作人的报告保存为txt文件，通过执行python代码写入到目录 "/Users/gitsilence/PycharmProjects/ai-study-project/autogen_/prompts/doc"
5、介绍下周树人和周作人之间的关系，以及经历，生成一个报告
6、将周树人和周作人之家的关系经历报告保存为txt文件，通过执行python代码保存到本地目录 "/Users/gitsilence/PycharmProjects/ai-study-project/autogen_/prompts/doc"
    """))
    pass

if __name__ == '__main__':
    import asyncio
    asyncio.run(run())
