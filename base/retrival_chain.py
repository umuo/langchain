from langchain_community.llms import Ollama
from langchain_community.document_loaders import WebBaseLoader
from langchain_community.embeddings import OllamaEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_core.documents import Document
from langchain.chains import create_retrieval_chain
from langchain_core.prompts import ChatPromptTemplate

# 第一步：初始化模型对象：LLM，Embedding
llm = Ollama(model="qwen2")
embeddings = OllamaEmbeddings(model="qwen2")
response1 = llm.invoke("langsmith是做什么的？")
print(f"检索前：{response1}")
# 第二步：获取数据
loader = WebBaseLoader("https://docs.smith.langchain.com/user_guide")
docs = loader.load()

# 第三步：文本拆分，文本Vector化
text_splitter = RecursiveCharacterTextSplitter()
documents = text_splitter.split_documents(docs)
# 使用 FAISS 将文档转化为向量，并创建一个向量数据库
# documents 是之前通过文本分割器得到的一组较小的文本片段。
# embeddings 是一个嵌入模型，用于将文本片段转换为向量表示。在这里，使用的是 OllamaEmbeddings 模型
vector = FAISS.from_documents(documents, embeddings)
"""
from_documents 方法会对 documents 进行迭代，使用 embeddings 模型将每个文档转换为向量。
然后，这些向量被添加到一个 FAISS 索引中，构建一个向量数据库。
最后，这个向量数据库存储在 vector 变量中，可以用于后续的相似性搜索和检索任务。
"""

# 第四步：准备Prompt，创建文档处理的LLMChain，检索Chain
prompt = ChatPromptTemplate.from_template("""Answer the following question based only on the provided context:
<context>
{context}
</context>
Question: {input}""")

# 创建处理文档链式调用
document_chain = create_stuff_documents_chain(llm, prompt)

# 将向量数据库转换为一个检索器对象，执行相似性搜索和文档检索操作
retriever = vector.as_retriever()

# 检索链条，用于结合向量检索和文档处理
retrieval_chain = create_retrieval_chain(retriever, document_chain)

# 第五步：代码执行与测试
response2 = document_chain.invoke({
    "input": "langsmith是做什么的？",
    "context": [Document(page_content="langsmith can let you visualize test results")]
})
print(f"检索前（运行测试）：{response2}")

response3 = retrieval_chain.invoke({"input": "langsmith是做什么的？"})
print(f"检索后：{response3['answer']}")
