# --coding: utf-8--
from mcp.server.fastmcp import FastMCP

# 创建一个MCP服务端
mcp = FastMCP(name="test_add_func",)


# 计算加法 的工具
@mcp.tool()
def add(a: int, b: int) -> int:
    return a + b


# 获取动态的资源信息
@mcp.resource("greeting://{name}")
def get_greeting(name: str) -> str:
    return f"Hello, {name}!"


# mcp.run(transport="sse")
