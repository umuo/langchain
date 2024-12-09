# --coding: utf-8--
"""
意图识别，并调用对应意图的function
"""
import sys, os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from base_llm import llm
from langchain.schema.runnable import RunnableSequence
from langchain.prompts import PromptTemplate
import re
import jinja2

# 1. 定义类别和对应的查询函数
categories = {
    "查询多次入所的未成年人": {
        "description": "查询未成年人多次入所记录",
        "query_function": "query_multiple_entry_minors"
    },
    "根据“籍贯|手机应用|足迹”查询人员": {
        "description": "查询天台本地人，手机安装过特定应用，最近去过国清寺的记录",
        "query_function": "query_by_location_app_footprint"
    },
    "有过涉黄和涉毒的嫌疑人": {
        "description": "查询有过涉黄和涉毒的嫌疑人记录",
        "query_function": "query_suspects_by_offense_type"
    }
}


# 2. 假设的查询函数（你需要根据你的实际数据源和查询方式实现这些函数）
def query_multiple_entry_minors(query):
    # 这里替换为实际的查询逻辑
    print(f"执行查询：{query} - 未成年人多次入所")
    return "查询结果：未成年人多次入所记录..."

def query_by_location_app_footprint(query, app_name, location, footprint=None):
    """
    根据地点、应用名称、人物关系和最近访问地点执行查询。
    Args:
        app_name: 应用名称。
        location: 地点。
        footprint: 最近访问过的地点，可选。

    Returns:
        查询结果。
    """

    description = f"执行查询：{query} - "

    if location and location != 'None':
        description += f"位于{location}，"

    if app_name and app_name != 'None':
        description += f"手机安装过{app_name}"

    if footprint and footprint != 'None':
        description += f"，最近去过{footprint}"

    print(description)
    # 这里替换为实际的查询逻辑，根据参数构建查询语句
    return f"查询结果：{description}的记录..."

def query_suspects_by_offense_type(query):
    # 这里替换为实际的查询逻辑
    print(f"执行查询：{query} - 有过涉黄和涉毒的嫌疑人")
    return "查询结果：有过涉黄和涉毒的嫌疑人记录..."

# 3. 使用 LLM 进行意图识别

intent_recognition_template = """
你是一个专业的意图识别助手。你的任务是根据用户的输入，判断其意图属于以下哪个类别，并只返回类别名称。

类别：
{% for category, details in categories.items() -%}
- {{ category }}：{{ details.description }}
{% endfor %}

用户输入：{user_input}

请只输出类别名称，不要输出其他任何内容。
"""
template = jinja2.Environment().from_string(intent_recognition_template)
intent_recognition_template = template.render({"categories": categories})
# print(intent_recognition_template)

intent_recognition_prompt = PromptTemplate(
    template=intent_recognition_template,
    input_variables=["user_input"]
)

intent_recognition_chain = RunnableSequence(
    intent_recognition_prompt | llm,
)

# 4. 提取参数（以 "天台本地人，手机安装过连信，最近去过国清寺" 为例）
parameter_extraction_template = """
用户输入：{user_input}
类别：{category}

从用户输入中提取以下参数：
- location: 用户的籍贯，籍贯是地区。像湖州人，籍贯就是湖州。如果没有提到则返回 "None"
- app_name: 用户手机安装的应用名称，如果没有提到则返回 "None"
- footprint: 用户最近去过的地点，如果没有提到则返回 "None"

请返回提取到的参数值，格式为：
location: <提取到的籍贯>
app_name: <提取到的应用名称>
footprint: <提取到的地点>
"""

parameter_extraction_prompt = PromptTemplate(
    template=parameter_extraction_template,
    input_variables=["user_input", "category"]
)

parameter_extraction_chain = RunnableSequence(
    parameter_extraction_prompt | llm
)

# 5. 主流程
def process_user_input(user_input):
    # 识别意图
    recognized_category = intent_recognition_chain.invoke({"user_input": user_input})
    print(recognized_category)
    if type(recognized_category) is dict:
        recognized_category = recognized_category['text']
    # 检查识别的类别是否存在
    if recognized_category not in categories:
        return "无法识别的类别。"

    parameters = {}
    if recognized_category == "根据“籍贯|手机应用|足迹”查询人员":
        extracted_params_str = parameter_extraction_chain.invoke(
            {"user_input": user_input, "category": recognized_category})
        # print("extracted_params_str: ", extracted_params_str)
        if type(extracted_params_str) is dict:
            extracted_params_str = extracted_params_str['text']
        # 使用正则表达式提取参数值
        location_match = re.search(r"location: (.*)", extracted_params_str)
        app_match = re.search(r"app_name: (.*)", extracted_params_str)
        footprint_match = re.search(r"footprint: (.*)", extracted_params_str)

        if location_match:
            parameters["location"] = location_match.group(1).strip()
        if app_match:
            parameters["app_name"] = app_match.group(1).strip()
        if footprint_match:
            parameters["footprint"] = footprint_match.group(1).strip()

        # print("regex match result: ", parameters)
        # 检查参数是否全部为空
        if not parameters:
            print("错误：至少需要提供籍贯、手机应用或足迹中的一个查询条件。")
            # 在这里可以根据你的需求进行错误处理，例如：
            # 1. 抛出异常
            # 2. 返回默认值或空结果
            # 3. 重新提示用户输入
            # ...

    # 执行查询
    query_function_name = categories[recognized_category]["query_function"]
    query_function = globals()[query_function_name]  # 根据函数名获取函数对象

    if recognized_category == "根据“籍贯|手机应用|足迹”查询人员":
        # 传入所有提取的参数，由查询函数内部处理空值情况
        result = query_function(user_input, **parameters)
    else:
        result = query_function(user_input)
    return result


# 6. 测试
user_inputs = [
    "帮我查一下多次入所的未成年人",
    "天台人，装了连信，去过国清寺的有哪些",
    "涉黄涉毒的嫌疑人有哪些？",
    "装了抖音的有哪些人",
    "今天天气怎么样？"
]

for user_input in user_inputs:
    print(f"用户输入：{user_input}")
    result = process_user_input(user_input)
    print(result)
    print("-" * 20)
