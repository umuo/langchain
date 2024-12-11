# --coding: utf-8--
from base.base_llm import llm
from langchain.chains import LLMChain
from langchain.prompts import PromptTemplate
from langchain.chains import SequentialChain

# 第一个 Chain：生成一个故事概要
first_prompt = PromptTemplate(
    input_variables=["topic"],
    template="写一个关于{topic}的简短故事大纲。"
)
chain_one = LLMChain(llm=llm, prompt=first_prompt, output_key="outline")

# 第二个 Chain：基于概要扩展成完整故事
second_prompt = PromptTemplate(
    input_variables=["outline"],
    template="基于以下大纲扩展成一个完整的故事：\n\n{outline}"
)
chain_two = LLMChain(llm=llm, prompt=second_prompt, output_key="story")

# 组合成 SequentialChain
overall_chain = SequentialChain(
    chains=[chain_one, chain_two],
    input_variables=["topic"],
    output_variables=["outline", "story"]
)

# 执行链
result = overall_chain({"topic": "程序员的一天"})
print("故事大纲:")
print(result["outline"])
print("\n完整故事:")
print(result["story"])

# LangChainDeprecationWarning: The class `LLMChain` was deprecated in LangChain 0.1.17 and will be removed in 1.0. Use :meth:`~RunnableSequence, e.g., `prompt | llm`` instead.
# LangChainDeprecationWarning: The method `Chain.__call__` was deprecated in langchain 0.1.0 and will be removed in 1.0. Use :meth:`~invoke` instead.