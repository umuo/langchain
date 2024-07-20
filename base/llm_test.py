from langchain_community.llms import Ollama
from langchain_core.prompts import ChatPromptTemplate

llm_data = {
    "model": "qwen2",
    "temperature": 0,
}
llm = Ollama(**llm_data)

# 创建一个对话提示模板
prompt = ChatPromptTemplate.from_messages([
    ("system", "现在你的名字叫小王，你来自M78星云"),
    ("user", "{input}")
])
tex = {"input": "你是哪位"}

# 这里的 | 是langchain自己实现的，用来做链式调用的。实际Python应用上还是运算符
# 在Python中可以通过定义类的特殊方法重载运算符，例如：| 运算符可以通过自定义 __or__ 方法来重载
# 这里是隐式调用
chain = prompt | llm
resp = chain.invoke({"input": "你叫什么名字"})
print(resp)
