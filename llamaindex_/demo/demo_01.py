import os
import sys

# 将当前脚本的父目录的父目录添加到 Python 的搜索路径中
# 这样可以导入父目录的父目录中的模块，例如这里的 base.base_llm
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# 从 base.base_llm 模块中导入 llm (可能是一个自定义的大语言模型实例)
from base.base_llm import llm
# 从 llama_index.embeddings.ollama 模块中导入 OllamaEmbedding 类
# 用于创建基于 Ollama 的文本嵌入模型
from llama_index.embeddings.ollama import OllamaEmbedding
# 从 llama_index.core 模块中导入 SimpleDirectoryReader 类
# 用于读取目录中的文件并加载为文档
from llama_index.core import SimpleDirectoryReader
# 从 llama_index.core 模块中导入 VectorStoreIndex, StorageContext, load_index_from_storage 类
# 用于构建向量存储索引、管理存储上下文以及从存储加载索引
from llama_index.core import VectorStoreIndex, StorageContext, load_index_from_storage

# 创建一个 OllamaEmbedding 实例，用于生成文本嵌入
embedding_model = OllamaEmbedding(
    model_name="qwen2:7b",  # 指定使用的 Ollama 模型名称
    base_url="http://10.118.21.135:11434",  # 指定 Ollama 服务的 URL
)

# 指定要处理的数据文件路径
data_dir = r"E:\DeskTop\tmp\1.txt"
# 使用 SimpleDirectoryReader 加载指定文件中的数据
documents = SimpleDirectoryReader(input_files=[data_dir]).load_data()
# 打印加载的文档数量和数据目录
print(f"Loaded {len(documents)} documents from {data_dir}")

# 使用 VectorStoreIndex.from_documents 方法构建向量存储索引
# documents: 要索引的文档列表
# embed_model: 用于生成文档嵌入的嵌入模型
index = VectorStoreIndex.from_documents(documents, embed_model=embedding_model)
# 打印索引构建成功的消息
print("Index built successfully")

# 指定索引保存的目录
index_dir = "index_data"
# 将索引的存储上下文持久化到指定目录
index.storage_context.persist(index_dir)
# 打印索引保存的目录
print(f"Index saved to {index_dir}")

# 示例查询部分
# 从指定目录加载存储上下文
storage_context = StorageContext.from_defaults(persist_dir=index_dir)
# 从存储上下文加载索引，并指定嵌入模型
loaded_index = load_index_from_storage(storage_context, embed_model=embedding_model)

# 使用加载的索引创建查询引擎，并指定大语言模型 (llm)
query_engine = loaded_index.as_query_engine(llm=llm)

# 定义要查询的问题
query_text = "公司总共赔了小杨多少钱？"
# 使用查询引擎执行查询
response = query_engine.query(query_text)
# 打印查询结果
print("\nQuery Results:")
print(f"Question: {query_text}")
print(f"Answer: {response}")
