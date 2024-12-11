# --coding: utf-8--
from base.base_llm import llm
from langchain.schema.runnable import RunnableSequence, RunnableParallel
from langchain.prompts import PromptTemplate

# 第一个 Chain：生成一个故事概要
first_prompt = PromptTemplate(
    input_variables=["topic"],
    template="写一个关于{topic}的简短故事大纲。"
)

# 第二个 Chain：基于概要扩展成完整故事
second_prompt = PromptTemplate(
    input_variables=["outline"],
    template="基于以下大纲扩展成一个完整的故事：\n\n{outline}"
)

# 创建 RunnableSequence
# overall_chain = RunnableSequence(
#     first_prompt, llm, lambda x: {"outline": x},
#     second_prompt, llm, lambda x: {"story": x}
# )

outline_chain = RunnableSequence(
    first_prompt, llm
)
story_chain = RunnableSequence(
    second_prompt, llm
)
# overall_chain = RunnableSequence(
#     {
#         "topic": lambda x: x["topic"]
#     },
#     {
#         "outline": outline_chain
#     },
#     {
#         "outline": lambda x: x["outline"],
#         "story": lambda x: story_chain.invoke({"outline": x["outline"]})
#     }
# )

overall_chain = RunnableSequence(
    {
        "topic": lambda x: x["topic"],
        "outline": outline_chain
    },
    {
        "topic": lambda x: x["topic"],
        "outline": lambda x: x["outline"],
        "story": lambda x: story_chain.invoke({"outline": x["outline"]})
    }
)
# 最后一个字典是result

# 执行序列
result = overall_chain.invoke({"topic": "程序员的一天"})
print(result)
print("故事大纲:")
print(result["outline"])
print("\n完整故事:")
print(result["story"])
