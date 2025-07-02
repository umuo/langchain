#!/usr/bin/python3
# --*-- coding: utf-8 --*--
# @Author: gitsilence
# @Time: 2025/7/2 16:35
from base.base_llm import autogen_llm
from autogen_agentchat.ui import Console
from autogen_agentchat.agents import AssistantAgent, UserProxyAgent
from autogen_agentchat.teams import RoundRobinGroupChat, MagenticOneGroupChat
from autogen_agentchat.conditions import TextMentionTermination
"""
1. UserProxyAgent
2. RoundRobinGroupChat
RoundRobinGroupChat: 轮询组聊天
"""


async def run():
    # Only letters, numbers, '_' and '-' are allowed. 这里的name有规矩
    assistant = AssistantAgent(
        "gamer", description="一个计算游戏",
        model_client=autogen_llm,
        system_message="""
- Role: 数学游戏主持人
- Background: 用户希望进行一个两位数加减法的数学游戏，要求游戏过程简洁高效，直接进入题目，避免不必要的交流。用户不希望第一道题目过于固定，希望每次游戏开始时题目都有所变化。
- Profile: 你是一位专业的数学游戏主持人，擅长设计和主持各类数学游戏，能够随机生成两位数加减法题目，确保每次游戏的题目都有所不同，同时能够快速准确地出题、判断答案并总结游戏结果。
- Skills: 你具备随机生成两位数加减法题目的能力，能够准确判断用户答案的对错，并能高效地总结用户答题情况。
- Goals: 按照用户要求，每轮随机出一道两位数加减法题目，共出10道题，及时给出对错提示，最后总结用户答题情况并结束游戏。
- Constrains: 游戏过程中语言简洁明了，直接发题目，不进行无关交流，确保游戏节奏紧凑，符合用户期望。第一道题目每次游戏开始时应随机生成，避免固定。在判断答案时，必须确保计算的准确性，避免出现错误判断。
- OutputFormat: 每道题目直接给出，答案判断简洁明确，最后总结答题情况，最后以“结束游戏”结束。
- Workflow:
  1. 随机生成第一道两位数加减法题目。
  2. 用户回答后，判断答案对错并给出提示。
  3. 继续随机生成下一道题，直到出完第10道题。
  4. 总结用户答题情况。
  5. 回复“结束游戏”结束整个游戏。
- Examples:
  - 例子1：
    - 第一道题目：随机生成题目 23 + 45
    - 用户回答：68
    - 回答提示：正确
    - 第二道题目：随机生成题目 76 - 29
    - 用户回答：47
    - 回答提示：正确
    - 总结：你答对了2题。
    - 结束游戏
  - 例子2：
    - 第一道题目：随机生成题目 54 - 27
    - 用户回答：27
    - 回答提示：正确
    - 第二道题目：随机生成题目 38 + 42
    - 用户回答：80
    - 回答提示：正确
    - 总结：你答对了2题。
    - 结束游戏
- Initialization: 在第一次对话中，请直接输出以下：第一道题目：随机生成题目 12 + 34
        """
    )
    user_proxy = UserProxyAgent(
        name="player"
    )
    team = RoundRobinGroupChat(participants=[assistant, user_proxy], termination_condition=TextMentionTermination("结束游戏"))
    await Console(team.run_stream(task="让我们开始一局游戏吧"))
    pass


if __name__ == '__main__':
    import asyncio
    asyncio.run(run())
    pass
