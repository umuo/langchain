#!/usr/bin/python3
# --*-- coding: utf-8 --*--
# @Author: gitsilence
# @Time: 2025/4/24 10:14
import asyncio
from langchain_openai import ChatOpenAI
from dotenv import load_dotenv
import os
from browser_use import Agent
import logging
logging.basicConfig(level=logging.DEBUG)
load_dotenv("../.env.local", override=True)


async def main():
    llm = ChatOpenAI(
        model_name=os.getenv("CHAT_MODEL_NAME"),
        api_key=os.getenv("OPENAI_API_KEY"),
        base_url=os.getenv("OPENAI_BASE_URL"),
        verbose=True
    )
    agent = Agent(
        task="今天的国际金价和国内金价分别是多少？",
        llm=llm,
        use_vision=False
    )
    await agent.run()


if __name__ == '__main__':
    asyncio.run(main())


