# --coding: utf-8--
from base.base_llm import llm
# from langchain.chains import LLMChain  # api废弃，但还可用
from langchain.schema.runnable import RunnableSequence
from langchain.prompts import PromptTemplate

prompt = PromptTemplate(template="你是一名资深学家，你涉及各个领域，你可以解答任何问题，请解答：{input}",
                        input_variables=["input"])
prompt2 = PromptTemplate(template="{summary}。 请根据以上描述帮我命名一个合适的名字，我的目录为：蜡笔小新第一季/1-xxx.mp4", input_variables=["summary"])

# RunnableSequence 允许您将多个组件（如提示模板、语言模型、输出解析器等）串联在一起，形成一个可执行的序列。
chain = RunnableSequence(prompt, llm,
                         lambda x: {"summary": x},  # 将第一次调用的结果传递给第二个prompt
                         prompt2, llm)
response = chain.invoke({"input": "emby家庭影院，我的资源如果想被它正常的刮削，我该怎么命名？"})
print(response)
