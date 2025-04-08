from langgraph import graph
from typing_extensions import TypedDict, Annotated
from langgraph.graph import StateGraph, START, END
from langgraph.graph.message import add_messages
from base.base_llm import llm
import asyncio
from dotenv import load_dotenv
load_dotenv("../.env.local")

# from IPython.display import Image, display

class State(TypedDict):
    # message的类型是list，通过add_messages函数将消息添加到messages
    messages: Annotated[list, add_messages]


def chatbot(state: State) -> dict:
    return {"messages": llm.invoke(state["messages"])}


# 将State作为输入，返回一个包含messages键 messages下更新列表的字典，这是langgraph节点函数的基本模式
graph_builder = StateGraph(State)
graph_builder.add_node("chatbot", chatbot)
# 添加entry point 告诉我们的graph从哪开始
graph_builder.add_edge(START, "chatbot")
# 同样设置一个finish point
graph_builder.add_edge("chatbot", END)

# 最后运行我们的graph，compile函数返回一个CompileGraph
graph = graph_builder.compile()


async def stream_graph_updates(user_input: str):
    import json
    events = [event async for event in graph.astream_events({"messages": [("user", user_input)]}, version="v2")]
    for event in events:
        # print(event)
        if event["event"] == "on_chain_stream" and event["name"] == "chatbot":
            print("AI:", event["data"]["chunk"]['messages'])
    # for event in graph.stream({"messages": [("user", user_input)]}):
    # for value in event.values():
    # print("AI:", value["messages"])


async def graph_updates(user_input: str):
    response = graph.invoke({"messages": [("user", user_input)]}, debug=True)
    print(response["messages"])
    for message in response["messages"]:
        print(message.content)


async def main():
    while True:
        try:
            user_input = input("用户: ")
            if user_input in ("exit", "quit", "q"):
                print("Goodbye!")
                break
            # await stream_graph_updates(user_input)
            await graph_updates(user_input)
        except Exception as e:
            print(f"Error: {e}")
            user_input = "你想知道关于langgraph的什么？"
            await stream_graph_updates(user_input)
            break


asyncio.run(main())
# try:
#     # get_graph您可以使用方法和其中一种“绘制”方法（例如draw_ascii或 ）来可视化图形draw_png 
#     # display(Image(graph.get_graph().draw_png()))
#     # print()
#     with open("flow.png", "wb") as f:
#         f.write(graph.get_graph().draw_mermaid_png())
# except:
#     pass
