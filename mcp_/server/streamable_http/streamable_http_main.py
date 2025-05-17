#!/usr/bin/python3
# --*-- coding: utf-8 --*--
# @Author: gitsilence
# @Time: 2025/5/13 15:45
from mcp.server.fastmcp import FastMCP
from mcp_.server.streamable_http.echo import mcp as echo
from mcp_.server.streamable_http.math import mcp as math
from fastapi import FastAPI


"""
可流式传输的 HTTP 
Streamable HTTP
"""
# lifespan 注册生命周期函数
app = FastAPI(lifespan=lambda app: echo.session_manager.run())
app.mount("/echo", echo.streamable_http_app())
app.mount("/math", math.streamable_http_app())

if __name__ == '__main__':
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8001)
