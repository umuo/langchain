from langchain.tools import Tool
"""
自定义工具
"""

def calculator_tool(input: str) -> str:
    try:
        print("tool input: ", input)
        return str(eval(input))  # 注意：eval 可能有安全问题，仅用于示例
    except Exception as e:
        print("tool error: ", e)
        return str(e)

calculator = Tool(
    name="Calculator",
    func=calculator_tool,
    description="用于执行简单的数学运算。输入格式为数学表达式，例如 '2 + 2'。",
)

