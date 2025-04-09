from langchain.prompts import PromptTemplate
from langchain.agents import ZeroShotAgent, Tool, AgentExecutor
from tools_01 import calculator
from langchain.callbacks.base import BaseCallbackHandler
from base.base_llm import llm
from langchain.chains import LLMChain
# ZeroShotAgent是一个基于提示模板的代理，用于生成响应。
class ScratchpadCallbackHandler(BaseCallbackHandler):
    def on_agent_action(self, action: dict, **kwargs):
        # action 是一个 AgentAction 对象，直接访问其 log 属性
        if hasattr(action, "log"):
            print(f"Agent Scratchpad at step:\n{action.log}\n")
        else:
            print("Action log is not available.")

# 创建 AgentExecutor 并附加回调
callback_handler = ScratchpadCallbackHandler()

prompt_template = ZeroShotAgent.create_prompt(
    tools=[calculator],
    prefix="You are an AI assistant, and you have access to the following tools:",  
    suffix="Begin your task by thinking step-by-step. If a tool call fails, analyze the reason and retry with a corrected input. \n Here is the question: {input}.\n\n {agent_scratchpad}",
    input_variables=["input", "agent_scratchpad"]
)
prompt_format = prompt_template.format(input="帮我计算 5 * (3 + 2 )", agent_scratchpad="")
print("prompt_format: ", prompt_format)

llm_chain = LLMChain(llm=llm, prompt=prompt_template)
# 创建ZeroShotAgent
agent = ZeroShotAgent(llm_chain=llm_chain, tools=[calculator])
# 创建AgentExecutor
agent_executor = AgentExecutor(agent=agent, tools=[calculator], callbacks=[callback_handler])

response = agent_executor.invoke({"input": "帮我计算 '5 * (3 + 2 )'"})
print(response)
