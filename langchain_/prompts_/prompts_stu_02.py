from langchain_core.runnables import RunnableLambda
import asyncio
import json
import time
from typing import AsyncIterator

# 1. 模拟长时间运行的任务
async def slow_process(text: str) -> AsyncIterator[str]:
    words = text.split()
    for word in words:
        # 模拟处理每个单词需要一些时间
        await asyncio.sleep(1)
        yield f"处理单词: {word}"

# 2. 带进度显示的处理器
async def process_with_progress(text: str) -> AsyncIterator[str]:
    total_steps = len(text.split())
    current_step = 0
    
    async for result in slow_process(text):
        current_step += 1
        progress = (current_step / total_steps) * 100
        yield f"{result} (进度: {progress:.1f}%)"

# 3. 模拟聊天机器人的打字效果
async def typing_effect(text: str) -> AsyncIterator[str]:
    for char in text:
        await asyncio.sleep(0.1)  # 模拟打字延迟
        yield char

# 创建可运行的链
slow_chain = RunnableLambda(slow_process)
progress_chain = RunnableLambda(process_with_progress)
typing_chain = RunnableLambda(typing_effect)

async def main():
    # 示例文本
    text = "这是一个测试句子 用来展示不同的效果"
    
    print("\n1. 监控长时间运行的任务:")
    events = [event async for event in slow_chain.astream_events(text, version="v2")]
    for event in events:
        if event["event"] == "on_chain_stream":
            print(event["data"]["chunk"])
    
    print("\n2. 带进度显示的任务:")
    events = [event async for event in progress_chain.astream_events(text, version="v2")]
    for event in events:
        if event["event"] == "on_chain_stream":
            print(event["data"]["chunk"])
    
    print("\n3. 聊天机器人打字效果:")
    response = "AI助手: 你好！我是一个AI助手，很高兴见到你！"
    start_time = time.time()
    print("", end="", flush=True)

    async for event in typing_chain.astream_events(response, version="v2"):
        if event["event"] == "on_chain_stream":
            print(event["data"]["chunk"], end="", flush=True)

    end_time = time.time()
    print(f"\n打印耗时: {end_time - start_time:.2f}秒", flush=True)
    print()  # 换行

if __name__ == "__main__":
    asyncio.run(main())
