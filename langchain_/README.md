
# Langchain

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
``