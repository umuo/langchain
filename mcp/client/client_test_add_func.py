# --coding: utf-8--
import types

from mcp.client.stdio import stdio_client
from mcp import ClientSession, StdioServerParameters, types
from pydantic import AnyUrl

server_params = StdioServerParameters(
    command="python",
    args=["../server/test_add_func.py"],
    env=None
)


async def handle_sampling_message(message: types.CreateMessageRequestParams) -> types.CreateMessageResult:
    return types.CreateMessageResult(
        role="assistant",
        content=[
            types.TextContent(
                type="text",
                text="Hello, world ! from model"
            )
        ],
        model="gpt-3.5-turbo",
        stopReason="endTurn"
    )


async def run():
    print("run: -> start !")
    async with stdio_client(server_params) as (read, write):
        session = ClientSession(read, write, sampling_callback=handle_sampling_message)
        await session.initialize()

        prompts = await session.list_prompts()
        print(f"prompts: \n -> {prompts}")

        prompt = await session.get_prompt("example-prompt", arguments={"a": "1"})
        print(f"prompt: \n -> {prompt}")

        resources = await session.list_resources()
        print(f"resources: \n -> {resources}")

        tools = await session.list_tools()
        print(f"tools: \n -> {tools}")

        content, mine_type = await session.read_resource(AnyUrl("greeting://zhangsan"))
        print(f"content: {content}, mine_type: {mine_type} ")

        result = await session.call_tool("test_add_func", arguments={"a": 1, "b": 2})
        print(f"result: \n -> {result}")


if __name__ == '__main__':
    import asyncio
    asyncio.run(run())
    pass
