from langchain_community.llms import Ollama
from langchain_core.prompts import ChatPromptTemplate

llm_data = {
    "model": "qwen2",
    "temperature": 0,
}
llm = Ollama(**llm_data)
context_list = [
    ("system", "现在你的名字叫小王，你来M78星云")
]
# 创建一个对话提示模板
# prompt = ChatPromptTemplate.from_messages([
#     ,
#     # ("user", "{input}")
# ])
# tex = {"input": "你是哪位"}

# 这里的 | 是langchain自己实现的，用来做链式调用的。实际Python应用上还是运算符
# 在Python中可以通过定义类的特殊方法重载运算符，例如：| 运算符可以通过自定义 __or__ 方法来重载
# 这里是隐式调用
# chain = prompt | llm
# resp = chain.invoke({"input": "你叫什么名字"})

# 下面是显式调用


def get_response(input_text):
    context_list.append(("user", input_text))
    prompt = ChatPromptTemplate.from_messages(context_list)
    # 生成响应
    resp = llm.stream(prompt.format())
    response_text = ""
    for chunk in resp:
        response_text += chunk
        print(chunk, end="", flush=True)
    print()
    return response_text


print("请输入你的对话内容，输入 'exit' 退出对话。")
while True:
    user_input = input("你: ")
    if user_input.lower() == "exit":
        break
    # 获取模型的响应
    response = get_response(user_input)
    # 将模型的响应也添加到上下文列表中
    context_list.append(('assistant', response))


# for resp_text in resp.generations[0]:
#     print(resp_text.text)
