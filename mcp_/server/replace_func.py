from mcp.server.lowlevel import Server, NotificationOptions
from mcp.server.models import InitializationOptions
import mcp.types as types
import mcp.server.stdio
import mcp.server.sse
from pydantic import FileUrl
import httpx
server = Server("test_replace_func")

"""
MCP 服务端示例
"""

SAMPLE_RESOURCES = {
    "greeting": "Hello! This is a sample text resource.",
    "help": "This server provides a few sample text resources for testing.",
    "about": "This is the simple-resource MCP server implementation.",
}


async def fetch_website(url: str) -> list[types.TextContent | types.ImageContent | types.EmbeddedResource]:
    print("fetching website: ", url)
    headers = {
        "User-Agent": "MCP Test Server (github.com/modelcontextprotocol/python-sdk)"
    }
    async with httpx.AsyncClient(follow_redirects=True, headers=headers) as client:
        response = await client.get(url)
        print(response.text)
        if response.status_code == 200:
            if response.headers["Content-Type"].startswith(("text/html", "text/plain", "application/json", "application/javascript")):
                return [types.TextContent(
                    type="text",
                    text=response.text
                )]
            elif response.headers["Content-Type"].startswith("image/"):
                import base64
                # 将图像内容转换为base64编码
                image_base64 = base64.b64encode(response.content).decode('utf-8')
                return [types.ImageContent(
                    type="image",
                    data=image_base64,
                    mimeType=response.headers["Content-Type"]
                )]
            else:
                # 处理其他内容类型，返回一个包含错误信息的文本内容
                return [types.TextContent(
                    type="text",
                    text=f"不支持的内容类型: {response.headers['Content-Type']}, {str(response.text)}"
                )]
        else:
            # 处理请求失败的情况
            return [types.TextContent(
                type="text",
                text=f"请求失败，状态码: {response.status_code}"
            )]


@server.list_tools()
async def list_tools() -> list[types.Tool]:
    return [
        types.Tool(
            name="get_web_content",
            description="这是用来获取网页中内容的工具",
            inputSchema={
                "type": "object",
                "required": ["url"],
                "properties": {
                    "url": {
                        "type": "string",
                        "description": "URL to fetch",
                    }
                },
            }
        ),
    ]


@server.call_tool()
async def fetch_web_tool(name: str, arguments: dict) -> list[types.TextContent | types.ImageContent | types.EmbeddedResource]:
    if name != "获取网页内容":
        raise ValueError(f"未找到对应的工具: {name}")
    if "url" not in arguments:
        raise ValueError("Missing required argument 'url'")
    url = arguments["url"]
    return await fetch_website(url)


@server.list_resources()
async def handle_list_resources() -> list[types.Resource]:
    return [
        types.Resource(
            uri=FileUrl(f"file:///{name}.txt"),
            name="示例资源~",
            description="这是一个示例资源!",
            type="text",
            content="这是一个示例资源@"
        )
        for name in SAMPLE_RESOURCES.keys()
    ]


@server.read_resource()
async def read_resource(uri: FileUrl) -> str | bytes:
    name = uri.path.replace(".txt", "").lstrip("/")
    if name not in SAMPLE_RESOURCES:
        raise ValueError(f"未找到对应的资源: {name}")
    return SAMPLE_RESOURCES[name]


@server.list_prompts()
async def handle_list_prompts() -> list[types.Prompt]:
    return [
        types.Prompt(
            name="示例prompt",
            description="这是一个示例prompt",
            arguments=[
                types.PromptArgument(
                    name="prompt-01",
                    description="这是第一个参数测试",
                    required=True,
                ),
                types.PromptArgument(
                    name="prompt-02",
                    description="这是第二个参数测试",
                    required=True,
                )
            ]
        ),
    ]


@server.get_prompt()
async def handle_get_prompt(
    name: str,
    arguments: dict[str, str] | None,
) -> types.GetPromptResult:
    print(arguments)
    if name != '示例prompt':
        raise ValueError(f"未找到对应的prompt: {name}")
    return types.GetPromptResult(
        description="示例prompt",
        messages=[
            types.PromptMessage(
                role="assistant",
                content=types.TextContent(
                    type="text",
                    text="你是一个乐于助人的助手！"
                )
            ),
            types.PromptMessage(
                role="user",
                content=types.TextContent(
                    type="text",
                    text=f"帮我写一个冒泡排序。收到的参数为：{arguments['prompt-01'] if 'prompt-01' in arguments else ''} -> {arguments['prompt-02'] if 'prompt-02' in arguments else ''}"
                )
            )
        ]
    )


async def run(transport: str):
    print("transport: ", transport)
    if transport != 'sse':
        async with mcp.server.stdio.stdio_server() as (read, write):
            await server.run(
                read_stream=read,
                write_stream=write,
                initialization_options=InitializationOptions(
                    server_name="example-server",
                    server_version="1.0.0",
                    capabilities=server.get_capabilities(
                        notification_options=NotificationOptions(),
                        experimental_capabilities={}
                    ),
                )
            )
    else:
        # sse
        sse_server = mcp.server.sse.SseServerTransport(endpoint="/messages/")
        from starlette.applications import Starlette
        from starlette.routing import Mount, Route

        async def handle_sse(request):
            async with sse_server.connect_sse(
                request.scope,
                request.receive,
                request._send
            ) as (read, write):
                await server.run(
                    read_stream=read,
                    write_stream=write,
                    initialization_options=InitializationOptions(
                        server_name="web_tool_server",
                        server_version="1.0.0",
                        capabilities=server.get_capabilities(
                            notification_options=NotificationOptions(),
                            experimental_capabilities={}
                        ),
                    )
                )

        starlette_app = Starlette(
            routes=[
                Route("/sse/", handle_sse),
                Mount("/messages/", app=sse_server.handle_post_message),
            ]
        )
        import uvicorn
        config = uvicorn.Config(starlette_app, host="127.0.0.1", port=8001)
        server_instance = uvicorn.Server(config=config)
        await server_instance.serve()

if __name__ == '__main__':
    import asyncio
    asyncio.run(run("sse"))
