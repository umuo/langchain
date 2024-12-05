# --coding: utf-8--
from base_llm import llm
from langchain.prompts import PromptTemplate

"""
使用PromptTemplate 类来创建一个简单的提示模板， 
然后使用 format 方法来填充模板中的变量，
最后调用 invoke 方法来生成响应
"""

template = "将后面的内容翻译成英文：{input}"

prompt = PromptTemplate(template=template, input_variables=["input"])
response = llm.invoke(prompt.format(input="你好吗？今天天气真不错！"))
print(response)  # How are you? The weather is really nice today!
