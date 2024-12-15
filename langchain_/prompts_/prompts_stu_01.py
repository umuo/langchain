from langchain_core.runnables import RunnableLambda
import asyncio
import json

async def reverse(s: str) -> str:
    return s[::-1]

chain = RunnableLambda(func=reverse)

async def main():
    # 方法1：使用 astream_events
    events = [event async for event in chain.astream_events("hello", version="v2")]
    print("Events:", json.dumps(events, indent=2))
    
    # 方法2：直接调用
    result = await chain.ainvoke("hello")
    print("Direct result:", result)


if __name__ == "__main__":
    asyncio.run(main())