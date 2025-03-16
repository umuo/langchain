from mcp import ClientSession, StdioServerParameters, types
from mcp.client.stdio import stdio_client
from pydantic import AnyUrl

"""
MCP 客户端调用示例
"""

server_params = StdioServerParameters(
    command="python",
    args=["/Users/gitsilence/PycharmProjects/ai-study-project/mcp/server/replace_func.py"],
)

async def handle_sampling_message(param):
    print(param)
    # return types.CreateMessageResult(
    #     role="assistant",
    #     content=types.TextContent(
    #         type="text",
    #         text="hello, world ! from model",
    #     ),
    #     model="gpt-3.5-turbo",
    #     stopReason="endTurn"
    # )


async def run():
    async with stdio_client(server_params) as (read, write):
        async with ClientSession(read, write, sampling_callback=handle_sampling_message) as session:
            # 初始化连接，返回服务端的版本信息、协议版本
            resp = await session.initialize()
            print(resp)
            print("----" * 20)
            
            # 列出所有的资源 resources
            resp = await session.list_resources()
            print(resp)
            print("----" * 20)

            # 列出所有的工具 tools
            resp = await session.list_tools()
            print(resp)
            print("----" * 20)

            # 列出所有的 prompts
            resp = await session.list_prompts()
            print(resp)
            print("====" * 20)

            # 读取资源
            resp = await session.read_resource(types.FileUrl(AnyUrl("file:///greeting.txt")))
            # 输出获取的资源结果
            print(resp.contents[0].text)

            print("====" * 20)
            # 调用工具
            resp = await session.call_tool(
                name="示例工具",
                arguments={"url": "https://blog.lacknb.cn/articles/random.json"}
            )
            # print(resp)

            # 获取Prompt
            resp = await session.get_prompt(name="示例prompt", arguments={"prompt-01": "1", "prompt-02": "2"})
            print(resp)



if __name__ == '__main__':
    import asyncio
    asyncio.run(run())
