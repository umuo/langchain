
from turtle import numinput
from langgraph import graph
from langgraph.graph import StateGraph
from typing_extensions import TypedDict, Annotated

# b 是int，或者是None
def reducer(a: list, b: int | None) -> list:
    """
    a是list，b是int，将b加入a
    """
    if b is not None:
        return a + [b]
    return a

# 定义状态类
class ChatStat(TypedDict):
    # x是一个列表类型，并且这个列表的内容经过reducer函数的处理
    x: Annotated[list, reducer]
    y: Annotated[list, reducer]

# 定义配置类
class ConfigSchema(TypedDict):
    index: int

# 定义节点函数   
def greet(state: ChatStat, config: ConfigSchema) -> dict:
    """
    这里返回必须是dict类型
    """
    # print(f"Received config: {config}")
    # print(f"Received state: {state}")
    num_list = state.get("x", [])
    num_index = config.get("index", 0)
    return {"x": num_list[num_index] ** 2}

# 创建状态图
graph = StateGraph(ChatStat, config_schema=ConfigSchema)
graph.add_node("greet", greet)
graph.set_entry_point("greet")
graph.set_finish_point("greet")

# print(ChatStat(x=123, y=456))

# 运行状态图
executor = graph.compile()
# print(executor.config_specs)
result = executor.invoke({"y": 0.2, "x": 0.9}, {"index": 0})

# {'x': [0.9, 0.81], 'y': [0.2]}
print(result)
