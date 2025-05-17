# Autogen 的使用
- 官方文档：https://microsoft.github.io/autogen/stable/user-guide/core-user-guide/installation.html
- Github仓库：https://github.com/microsoft/autogen

## 常用组件
### LLM组件
例如： `from autogen_ext.models.openai import OpenAIChatCompletionClient`
这是用来调用 OpenAI 的 LLM 模型的组件。
其他的组件有anthropic、azure、cache、llama_cpp、ollama、openai、replay

anthropic、azure、openai这几个是厂商的，ollama通过ollama的api

- [llama_cpp](https://github.com/ggml-org/llama.cpp) 可以用来加载本地模型，例如gguf格式的
- cache：可以和openai一块使用，将输入和输出缓存到本地文件，下次再输入相同的输入就返回缓存中的输出
- replay：自己mock几个llm输出列表，调用的时候就会按照这个列表中的内容输出，每次输入一个内容就对应输出一个内容

当我们使用 OpenAIChatCompletionClient 要调用非OpenAI的模型时，需要设置`model_info`，例如下面的代码：
```python
from autogen_ext.models.openai import OpenAIChatCompletionClient
from autogen_core.models import UserMessage
import asyncio

async def main():
    model_client = OpenAIChatCompletionClient(
        api_key="sk-xxxx",
        base_url="https://xx.com/v1",
        model="moonshot-v1-32b",
        model_info={
            "json_output": False,
            "vision": False,
            "function_calling": True,
            "family": "unknown",
            "structured_output": False,
        }
    )
    response = await model_client.create([UserMessage(
        content="你好",
        source="user",
    )])
    print(response)
if __name__ == '__main__':
    asyncio.run(main())
```
如果不设置`model_info`，调用非openai的模型就会报错

### Agent组件

#### AssistantAgent
支持普通回复、流式响应、工具调用、结构化输出、handoff（交接，多Agent协同）、
模型上下文管理（支持基于 token 或 message 数量的上下文截断策略（model_context 设为 BufferedChatCompletionContext 或 TokenLimitedChatCompletionContext）、
默认支持并发调用多个工具、支持workbench（MCP工作区）

不是一个线程安全，不应该在多个task中并发调用
```python
import asyncio
from autogen_agentchat.agents import AssistantAgent
from base.base_llm import autogen_llm


async def run():
    agent = AssistantAgent("assistant", model_client=autogen_llm)  # 实例化一个助手角色
    result = await agent.run(task="说 “你好世界！”")
    for message in result.messages:
        print(f"{message.source}: {message.content}")
    await autogen_llm.close()

if __name__ == "__main__":
    asyncio.run(run())
```
工具调用：
```python
#!/usr/bin/python3
# --*-- coding: utf-8 --*--
# @Author: gitsilence
# @Time: 2025/5/17 13:52
from base.base_llm import autogen_llm
from autogen_core.tools import FunctionTool
# """A token used to cancel pending async calls"""
from autogen_core import CancellationToken
from autogen_agentchat.agents import AssistantAgent
import asyncio


async def add(a: int, b: int) -> int:
    print(f"函数调用：{a} + {b}")
    return a + b


async def run():
    add_tool = FunctionTool(add, description="求两数之和")
    # 手动调用
    # result = await add_tool.run_json({"a": 1, "b": 2}, CancellationToken())
    assistant_agent = AssistantAgent("assistant", model_client=autogen_llm, tools=[add_tool])
    result = await assistant_agent.run(task="计算 983+331 的结果")
    # print(add_tool.return_value_as_string(result))
    for message in result.messages:
        print(f"{message.source}: {message.content}")
    pass
    await autogen_llm.close()

if __name__ == '__main__':
    asyncio.run(run())
```
```text
函数调用：983 + 331
user: 计算 983+331 的结果
assistant: [FunctionCall(id='add:0', arguments='{\n    "a": 983,\n    "b": 331\n}', name='add')]
assistant: [FunctionExecutionResult(content='1314', name='add', call_id='add:0', is_error=False)]
assistant: 1314
```
调用Mcp Workbench
```python
#!/usr/bin/python3
# --*-- coding: utf-8 --*--
# @Author: gitsilence
# @Time: 2025/5/17 14:06
"""
MCP Workbench
"""
import asyncio
from base.base_llm import autogen_llm
from autogen_agentchat.agents import AssistantAgent
from autogen_ext.tools.mcp import McpWorkbench, StdioServerParams

hot_topic_mcp = StdioServerParams(
    command="/Users/gitsilence/miniforge3/envs/machine_learning/bin/python",
    args=["/Users/gitsilence/PycharmProjects/hot-topic-mcp/main.py"]
)


async def run():
    async with McpWorkbench(hot_topic_mcp) as workbench:
        assistant_agent = AssistantAgent("assistant", model_client=autogen_llm, workbench=workbench)
        result = await assistant_agent.run(task="查一下当前新浪微博的热搜，取前5条")
        for message in result.messages:
            print(f"{message.source}: {message.content}")
        pass
    await autogen_llm.close()


if __name__ == '__main__':
    asyncio.run(run())
    pass
```
```text
user: 查一下当前新浪微博的热搜，取前5条
assistant: [FunctionCall(id='sina_weibo_hot_topic:0', arguments='{\n    "top": 5\n}', name='sina_weibo_hot_topic')]
assistant: [FunctionExecutionResult(content='[{"title": "28岁男子食物中毒离世", "url": "https://so.sina.cn/search/list.d.html?keyword=28%E5%B2%81%E7%94%B7%E5%AD%90%E9%A3%9F%E7%89%A9%E4%B8%AD%E6%AF%92%E7%A6%BB%E4%B8%96"}, {"title": "孙杨比潘展乐快了1秒多", "url": "https://so.sina.cn/search/list.d.html?keyword=%E5%AD%99%E6%9D%A8%E6%AF%94%E6%BD%98%E5%B1%95%E4%B9%90%E5%BF%AB%E4%BA%861%E7%A7%92%E5%A4%9A"}, {"title": "国家网络身份认证App保护信息安全", "url": "https://so.sina.cn/search/list.d.html?keyword=%E5%9B%BD%E5%AE%B6%E7%BD%91%E7%BB%9C%E8%BA%AB%E4%BB%BD%E8%AE%A4%E8%AF%81App%E4%BF%9D%E6%8A%A4%E4%BF%A1%E6%81%AF%E5%AE%89%E5%85%A8"}, {"title": "朱雀二号改进型遥二运载火箭成功发射", "url": "https://so.sina.cn/search/list.d.html?keyword=%E6%9C%B1%E9%9B%80%E4%BA%8C%E5%8F%B7%E6%94%B9%E8%BF%9B%E5%9E%8B%E9%81%A5%E4%BA%8C%E8%BF%90%E8%BD%BD%E7%81%AB%E7%AE%AD%E6%88%90%E5%8A%9F%E5%8F%91%E5%B0%84"}, {"title": "孙颖莎王楚钦等出战世乒赛", "url": "https://so.sina.cn/search/list.d.html?keyword=%E5%AD%99%E9%A2%96%E8%8E%8E%E7%8E%8B%E6%A5%9A%E9%92%A6%E7%AD%89%E5%87%BA%E6%88%98%E4%B8%96%E4%B9%92%E8%B5%9B"}]', name='sina_weibo_hot_topic', call_id='sina_weibo_hot_topic:0', is_error=False)]
assistant: [{"title": "28岁男子食物中毒离世", "url": "https://so.sina.cn/search/list.d.html?keyword=28%E5%B2%81%E7%94%B7%E5%AD%90%E9%A3%9F%E7%89%A9%E4%B8%AD%E6%AF%92%E7%A6%BB%E4%B8%96"}, {"title": "孙杨比潘展乐快了1秒多", "url": "https://so.sina.cn/search/list.d.html?keyword=%E5%AD%99%E6%9D%A8%E6%AF%94%E6%BD%98%E5%B1%95%E4%B9%90%E5%BF%AB%E4%BA%861%E7%A7%92%E5%A4%9A"}, {"title": "国家网络身份认证App保护信息安全", "url": "https://so.sina.cn/search/list.d.html?keyword=%E5%9B%BD%E5%AE%B6%E7%BD%91%E7%BB%9C%E8%BA%AB%E4%BB%BD%E8%AE%A4%E8%AF%81App%E4%BF%9D%E6%8A%A4%E4%BF%A1%E6%81%AF%E5%AE%89%E5%85%A8"}, {"title": "朱雀二号改进型遥二运载火箭成功发射", "url": "https://so.sina.cn/search/list.d.html?keyword=%E6%9C%B1%E9%9B%80%E4%BA%8C%E5%8F%B7%E6%94%B9%E8%BF%9B%E5%9E%8B%E9%81%A5%E4%BA%8C%E8%BF%90%E8%BD%BD%E7%81%AB%E7%AE%AD%E6%88%90%E5%8A%9F%E5%8F%91%E5%B0%84"}, {"title": "孙颖莎王楚钦等出战世乒赛", "url": "https://so.sina.cn/search/list.d.html?keyword=%E5%AD%99%E9%A2%96%E8%8E%8E%E7%8E%8B%E6%A5%9A%E9%92%A6%E7%AD%89%E5%87%BA%E6%88%98%E4%B8%96%E4%B9%92%E8%B5%9B"}]
```

#### UserProxyAgent


### Team 多Agent协作
