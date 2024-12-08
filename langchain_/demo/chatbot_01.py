"""
基于langchain实现的带有记忆的聊天机器人
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from base_llm import llm
from langchain.memory import ConversationBufferWindowMemory
from langchain.schema.runnable import RunnableSequence, RunnablePassthrough
from langchain.prompts import ChatPromptTemplate

# 只保留最近的5轮对话，防止上下文溢出
memory = ConversationBufferWindowMemory(k=10)

prompt = ChatPromptTemplate.from_template("""你是一个专业、友好且富有同理心的AI助手。你的回答应该：
1. 准确且有帮助
2. 简洁明了，避免冗长
3. 保持礼貌和专业性
4. 在合适的时候可以表达共鸣

以下是之前的对话历史：
{history}

请回答用户的问题。

用户：{input}
AI助手：""")

while True:
    input_text = input("用户: ")
    if input_text == "exit":
        break
    chain = RunnableSequence(
        {
            "history": lambda x: memory.load_memory_variables({})["history"],
            "input": RunnablePassthrough(),
        } | prompt | llm
    )
    response_text = ""
    for chunk in chain.stream(input_text):
        response_text += chunk
        print(chunk, end="", flush=True)
    print()
    
    print("上下文：", memory.load_memory_variables({}))
    memory.save_context({"input": input_text}, {"output": response_text})
