from langchain_core.runnables import RunnableLambda
import asyncio
import json

"""
在 LangChain 中有两个版本的事件流格式：

v1（默认）：较旧的格式，事件结构相对简单
v2：新版本格式，提供更详细的事件信息，包括：
run_id：唯一运行标识符
parent_ids：父运行的标识符（用于跟踪嵌套调用）
tags：事件标签
metadata：额外元数据
"""

async def simple_func(text: str) -> str:
    return text.upper()

chain = RunnableLambda(func=simple_func)

async def main():
    text = "Hello World"
    
    print("Version 1 events:")
    events = [event async for event in chain.astream_events(text, version="v1")]
    print(json.dumps(events, indent=2, ensure_ascii=False))
    
    print("\nVersion 2 events:")
    events = [event async for event in chain.astream_events(text, version="v2")]
    print(json.dumps(events, indent=2, ensure_ascii=False))

if __name__ == "__main__":
    asyncio.run(main())
