
# Langchain

https://python.langchain.com/api_reference/core/runnables/langchain_core.runnables.base.RunnableSequence.html
文档官网

## 核心组件

### PromptTemplate
用于动态生成对LLM的输入提示

```python
from langchain.prompts.chat import ChatPromptTemplate
from langchain.llms import OpenAI
llm = OpenAI(temperature=0)
prompt = ChatPromptTemplate.from_template("你好，我是{input}")
response = llm(prompt.format(input="小王"))
print(response)
```

### Chains
将多个步骤组合在一起，构成一个流畅的处理管道，用于处理复杂的任务。

```python
from langchain.chains import LLMChain
from langchain.llms import OpenAI
llm = OpenAI(temperature=0)
prompt = ChatPromptTemplate.from_template("你好，我是{input}")
chain = LLMChain(llm=llm, prompt=prompt)
response = chain.run("小王")
print(response)
```
上面的API在新版中已经被弃用

```python
from langchain.schema.runnable import RunnableSequence
from langchain.llms import OpenAI
llm = OpenAI(temperature=0)
prompt = ChatPromptTemplate.from_template("你好，我是{input}")
chain = RunnableSequence(prompt, llm)
response = chain.invoke({"input": "小王"})
print(response)
```

### Agents
动态决定使用那种工具和操作，可以处理开放式任务，包括搜索，问答，代码生成等。

```python
from langchain.agents import load_tools
from langchain.agents import initialize_agent
from langchain.llms import OpenAI
llm = OpenAI(temperature=0)
tools = load_tools(["serpapi", "llm-math"], llm=llm)
agent = initialize_agent(tools, llm, agent="zero-shot-react-description", verbose=True)
agent.run("What is the current price of Apple stock?")
```

### Memory
给应用程序提供上下文记忆，可以用于存储会话历史，以及为应用程序提供更多上下文信息，以便更好地理解用户的问题和生成更准确的回答。

```python
from langchain.memory import ConversationBufferMemory
from langchain.chains import LLMChain
from langchain.llms import OpenAI
llm = OpenAI(temperature=0)
memory = ConversationBufferMemory()
memory.save_context({"input": "小王"}, {"output": "你好，我是小王"})
memory.save_context({"input": "李四"}, {"output": "你好，我是李四"})

prompt = ChatPromptTemplate.from_template("你好，我是{input}")
chain = LLMChain(llm=llm, prompt=prompt, memory=memory)
response = chain.run("王二")
print(response)
```
上面的API在新版中已经被弃用

```python
from langchain.memory import ConversationBufferMemory
from langchain.schema.runnable import RunnableSequence, RunnablePassthrough
from langchain.llms import OpenAI
llm = OpenAI(temperature=0)

memory = ConversationBufferMemory()
memory.save_context({"input": "小王"}, {"output": "你好，我是小王"})
memory.save_context({"input": "李四"}, {"output": "你好，我是李四"})

prompt = ChatPromptTemplate.from_template("聊天内容上下文如下：{history}\n结合上下文回答用户的问题\n 用户：{input}")
chain = RunnableSequence(
    {
        "history": lambda x: memory.load_memory_variables({})["history"],
        "input": RunnablePassthrough(),
    } | prompt | llm
)
response = chain.invoke({"input": "王二"})
print(response)
```

## ReAct(Reasoning and Acting) 结合推理

ReAct 是一种结合推理（Reasoning）和行动（Acting）的策略，它的目标是让语言模型能够在复杂任务中既能分析问题、推导答案，又能调用工具或采取行动。其核心在于将思维链（Chain Of Thought, COT）和工具调用流程融合，通过交替使用思考（Thought）和行动（Action）来实现复杂任务的解决。

**ReAct的核心流程由以下几个步骤构成**：
1. 思考（Thought）
- 模型通过自我推理分析当前需要解决的问题
- 例如，“我需要计算总价，包括税费”

2. 行动（Action） 
- 在需要工具辅助时，选择调用工具计算输入。
- 例如，调用计算器工具，输入表达式 100 + 15

3. 观察（Observation）
- 工具返回结果后，模型获取结果并进行下一步推理

4. 重复（Repeat） 
- 上述流程可以循环多次，直到问题解决或达到指定的终止条件。

5. 最终答案（Final Answer） 
- 在所有模型推理和工具调用完成后，模型给出最终答案。

以下是 ReAct 的典型Prompt模板：
```plaintext
You are an intelligent assistant and can think and act step by step to solve problems. 
You have access to tools. Here is how you interact:

Use the following format:
Question: the input question you must answer
Thought: think about the problem and determine the next step
Action: the action to take, choose from [Tool1, Tool2]
Action Input: the input for the action
Observation: the result of the action
... (this Thought/Action/Observation cycle can repeat N times)
Thought: I now know the final answer
Final Answer: the final answer to the original input question
```

**ReAct的优势：**
1. 动态推理（Dynamic Reasoning）
   - ReAct 支持多步推理，能够处理动态、复杂的任务，而不是一次性输出结果。
2. 灵活工具使用（Flexible Tool Usage） 
   - ReAct可以在需要调用工具，通过外部能力解决问题。
3. 错误修正能力（Error Correction）
   - 如果中途出现错误，ReAct可以通过观察工具反馈进行修正，尝试新的路径。


## CoT(Chain of Thought)
Chain Of Thought 是一种通过显式的推理步骤来解决问题的方法，它的思路是将问题分解成更小的子问题，然后依次进行推理，最终得到最终的答案。

**CoT的核心流程：**
1. 输入问题：
   - 模型接收到问题
2. 生成推理过程
   - 模型在回答问题前，逐步列出其逻辑推导过程。
3. 输出答案
   - 模型根据推理过程生成最终的答案

CoT的Prompt模板:
```plaintext
Question: How many people are on the bus?
Details: Initially, there are 10 people on the bus. At the first stop, 3 get off and 5 get on. At the second stop, 2 get off and 4 get on. How many people are on the bus now?

Let's think step by step:
```
CoT 的示例
问题: How many people are on the bus?

```plaintext
Question: How many people are on the bus?
Initially, there are 10 people on the bus.
At the first stop, 3 get off, so there are 10 - 3 = 7 people.
Then, 5 get on, so there are 7 + 5 = 12 people.
At the second stop, 2 get off, so there are 12 - 2 = 10 people.
Finally, 4 get on, so there are 10 + 4 = 14 people.
The final answer is 14.
```
首先公交车上默认有10个人，在第一站点有3人下车，有5人上车，在第二站点有2人下车，有4人上车，最终公交车上有14个人。
10 - 3 = 7
7 + 5 = 12
12 - 2 = 10
10 + 4 = 14
最终答案是14人。


ReAct 与 CoT 的区别和联系:

| 特性 | ReAct | CoT |
| --- | --- | --- |
| **目标** | 推理 + 调用工具，解决动态任务 | 推理，解决逻辑复杂问题 |
| **工具调用** | 支持调用外部工具 | 不调用工具，只靠推理 |
| **实现方式** | 交替进行“推理-行动-观察”循环 | 通过连续的逻辑推理得出答案 |
| **终止条件** | 观察工具结果，逐步调整，直至问题解决 | 推理过程结束后直接输出答案 |
| **示例问题** | 调用计算器解决数学表达式，或者多步动态任务 | 解决逻辑题、数学题、文本推理等问题 |


•	CoT：适合解决需要深度逻辑推理的问题，例如数学题或逻辑问题，旨在提高模型的推理透明度。
•	ReAct：整合了 CoT 的推理能力和工具调用的行动能力，适用于动态任务或需要外部工具支持的场景。