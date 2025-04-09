# Langchain 工具调用过程
前提是需要调用工具的LLM模型支持工具调用，例如qwen2:7b。

## 调试的代码
```python
#!/usr/bin/python3
# --*-- coding: utf-8 --*--
# @Author: gitsilence
# @Time: 2025/4/9 00:45
import os
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from langchain.agents import create_tool_calling_agent, AgentExecutor
from tools_01 import calculator_tool_02
from langchain_core.prompts import ChatPromptTemplate


load_dotenv('../../.env.local')
API_KEY = os.getenv("OPENAI_API_KEY")
BASE_URL = os.getenv("OPENAI_BASE_URL")
CHAT_MODEL = os.getenv("CHAT_MODEL_NAME")

llm = ChatOpenAI(
    model=CHAT_MODEL,
    api_key=API_KEY,
    base_url=BASE_URL
)
tools = [calculator_tool_02]
prompt = ChatPromptTemplate.from_messages([
    ("system", "你是一个数学助手"),
    ("user", "{input}"),
    ("placeholder", "{agent_scratchpad}"),
])
agent = create_tool_calling_agent(llm, tools, prompt)
agent_executor = AgentExecutor(agent=agent, tools=tools, verbose=True)
response = agent_executor.invoke({"input": "(9 * 9 - 2 * 2) / 7的结果是多少？"})
print(f"""
input: {response["input"]}
output: {response["output"]}
""")
```
- 输出结果
```text
Invoking: `calculator_tool_02` with `{'input': '(9 * 9 - 2 * 2) / 7'}`

tool invoke:  (9 * 9 - 2 * 2) / 7
11.0计算 `(9 * 9 - 2 * 2) / 7` 的结果是 `11.0`.

> Finished chain.

input: (9 * 9 - 2 * 2) / 7的结果是多少？
output: 计算 `(9 * 9 - 2 * 2) / 7` 的结果是 `11.0`.
```

## 具体研究
- 将工具封装请求体中，请求url接口
```json
{
  "model": "qwen2:7b",
  "stream": true,
  "tools": [
    {
      "type": "function",
      "function": {
        "name": "calculator_tool_02",
        "description": "用于执行简单的数学运算。输入格式为数学表达式，例如 '2 x 2'。",
        "parameters": {
          "properties": {
            "input": {
              "type": "string"
            }
          },
          "required": [
            "input"
          ],
          "type": "object"
        }
      }
    }
  ],
  "messages": [
    {
      "content": "你是一个数学助手",
      "role": "system"
    },
    {
      "content": "(9 * 9 - 2 * 2) / 7的结果是多少？",
      "role": "user"
    }
  ]
}
```
- 接口返回结果
如果需要调用工具，会返回tool_calls字段，其中包含工具名称和参数。
否则LLM就直接返回结果。
```json
{
  "id": "chatcmpl-850",
  "choices": [
    {
      "delta": {
        "content": "",
        "function_call": null,
        "refusal": null,
        "role": "assistant",
        "tool_calls": [
          {
            "index": 0,
            "id": "call_vxxq5u1i",
            "function": {
              "arguments": "{\"input\":\"(9 * 9 - 2 * 2) / 7\"}",
              "name": "calculator_tool_02"
            },
            "type": "function"
          }
        ]
      },
      "finish_reason": null,
      "index": 0,
      "logprobs": null
    }
  ],
  "created": 1744165949,
  "model": "qwen2:7b",
  "object": "chat.completion.chunk",
  "service_tier": null,
  "system_fingerprint": "fp_ollama",
  "usage": null
}
```

- Langchain代码处理tool_calls
`langchain.agents.agent.AgentExecutor._perform_agent_action`
```python
def _perform_agent_action(
    self,
    name_to_tool_map: Dict[str, BaseTool],
    color_mapping: Dict[str, str],
    agent_action: AgentAction,
    run_manager: Optional[CallbackManagerForChainRun] = None,
) -> AgentStep:
    if run_manager:
        run_manager.on_agent_action(agent_action, color="green")
    # Otherwise we lookup the tool
    if agent_action.tool in name_to_tool_map:
        tool = name_to_tool_map[agent_action.tool]
        return_direct = tool.return_direct
        color = color_mapping[agent_action.tool]
        tool_run_kwargs = self._action_agent.tool_run_logging_kwargs()
        if return_direct:
            tool_run_kwargs["llm_prefix"] = ""
        # We then call the tool on the tool input to get an observation
        observation = tool.run(
            agent_action.tool_input,
            verbose=self.verbose,
            color=color,
            callbacks=run_manager.get_child() if run_manager else None,
            **tool_run_kwargs,
        )
    else:
        tool_run_kwargs = self._action_agent.tool_run_logging_kwargs()
        observation = InvalidTool().run(
            {
                "requested_tool_name": agent_action.tool,
                "available_tool_names": list(name_to_tool_map.keys()),
            },
            verbose=self.verbose,
            color=None,
            callbacks=run_manager.get_child() if run_manager else None,
            **tool_run_kwargs,
        )
    return AgentStep(action=agent_action, observation=observation)
```
如果工具存在，就调用工具，否则调用InvalidTool。

- 工具调用结果返回给LLM
```json
{
  "model": "qwen2:7b",
  "stream": true,
  "tools": [
    {
      "type": "function",
      "function": {
        "name": "calculator_tool_02",
        "description": "用于执行简单的数学运算。输入格式为数学表达式，例如 '2 x 2'。",
        "parameters": {
          "properties": {
            "input": {
              "type": "string"
            }
          },
          "required": [
            "input"
          ],
          "type": "object"
        }
      }
    }
  ],
  "messages": [
    {
      "content": "你是一个数学助手",
      "role": "system"
    },
    {
      "content": "(9 * 9 - 2 * 2) / 7的结果是多少？",
      "role": "user"
    },
    {
      "content": null,
      "role": "assistant",
      "tool_calls": [
        {
          "type": "function",
          "id": "call_4tfguh7k",
          "function": {
            "name": "calculator_tool_02",
            "arguments": "{\"input\": \"(9 * 9 - 2 * 2) / 7\"}"
          }
        }
      ]
    },
    {
      "content": "11.0",
      "role": "tool",
      "tool_call_id": "call_4tfguh7k"
    }
  ]
}
```
- 最后LLM返回结果
```text
input: (9 * 9 - 2 * 2) / 7的结果是多少？
output: 计算 `(9 * 9 - 2 * 2) / 7` 的结果是 `11.0`.
```