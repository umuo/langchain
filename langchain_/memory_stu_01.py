# --coding: utf-8--
from base_llm import llm
from langchain.prompts import ChatPromptTemplate
from langchain.memory import ConversationBufferMemory
from langchain.schema.runnable import RunnableSequence, RunnablePassthrough
"""
用于存储完整上下文会话
"""
memory = ConversationBufferMemory()
memory.save_context({"input": "你是谁"}, {"output": "我是一只狗。"})
memory.save_context({"input": "你擅长什么？"}, {"output": "我擅长狗叫。"})

print(memory.load_memory_variables({}))

prompt = ChatPromptTemplate.from_template("聊天上下文如下：\n{history}\n结合上下文回答用户的问题。用户：{input}\nAI：")

chain = RunnableSequence(
    {
        "history": lambda x: memory.load_memory_variables({})["history"],
        "input": RunnablePassthrough(),
    } | prompt | llm
)
response = chain.invoke("你擅长什么？")
print(response)


