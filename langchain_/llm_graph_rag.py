# --coding: utf-8--
from langchain_community.graphs.nebula_graph import NebulaGraph
from langchain.chains import NebulaGraphQAChain
from langchain_openai import ChatOpenAI
from dotenv import load_dotenv
# 按优先级依次加载配置文件
load_dotenv('.env', override=True)            # 最低优先级
load_dotenv('.env.local', override=True)      # 最高优先级


llm = ChatOpenAI(
    model="moonshot-v1-8k",
    api_key="sk-clHCmfujSb1xvjOfQpB7Ojp7VtWfZVHmYjaGp8o0xdFfvz3Y",
    base_url="http://10.118.21.135:3000/v1"
)
graph_db = NebulaGraph(
    space="nezha2",
    username="root",
    password="nebula",
    address="10.118.21.49",
    port=9669,
)
graph_db.refresh_schema()
print(graph_db.get_schema)

qa_chain = NebulaGraphQAChain.from_llm(
    llm=llm,
    graph_db=graph_db,
    verbose=True,
)
question = "介绍下哪吒的关系网"
result = qa_chain.run({
    "input": question
})
print(result)


