from langgraph.graph import StateGraph
from typing_extensions import TypedDict

# 定义状态类
class ChatStat(TypedDict):
    message: str

# 定义节点函数
def greet(state: ChatStat) -> dict:
    return {"message": f"Hello! You said: {state['message']}"}

# 创建状态图
graph = StateGraph(ChatStat)
graph.add_node("greet", greet)
graph.set_entry_point("greet")
graph.set_finish_point("greet")

# 运行状态图
executor = graph.compile()
result = executor.invoke(ChatStat(message="Hello World !"))
print(result["message"])
