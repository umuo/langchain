# --coding: utf-8--
from base_llm import llm
from langchain.prompts import ChatPromptTemplate
from langchain.memory import ConversationBufferMemory
from langchain.chains import LLMChain
"""
用于存储完整上下文会话
"""
memory = ConversationBufferMemory()
memory.save_context({"input": "你是谁"}, {"output": "我是一只狗。"})
memory.save_context({"input": "你擅长什么？"}, {"output": "我擅长狗叫。"})

print(memory.load_memory_variables({}))
prompt = ChatPromptTemplate.from_template("历史会话：{history}。\n结合上下文回答用户的问题。Human: {input}\nAI: ")

llm_chain = LLMChain(llm=llm, memory=memory, prompt=prompt, output_key="output")
response = llm_chain.invoke({"input": "把我上一个问题发出来"})
print(response)
print(response['output'])


