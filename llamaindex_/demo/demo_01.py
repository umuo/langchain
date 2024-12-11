import os
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from base.base_llm import llm
from llama_index.embeddings.ollama import OllamaEmbedding
from llama_index.core import SimpleDirectoryReader
from llama_index.core import VectorStoreIndex, StorageContext, load_index_from_storage

embedding_model = OllamaEmbedding(
    model_name="qwen2:7b",
    base_url="http://10.118.21.135:11434",
)

data_dir = r"E:\DeskTop\tmp\1.txt"
# 加载数据
documents = SimpleDirectoryReader(input_files=[data_dir]).load_data()
print(f"Loaded {len(documents)} documents from {data_dir}")

# 构建索引
index = VectorStoreIndex.from_documents(documents, embed_model=embedding_model)
print("Index built successfully")

# 保存索引
index_dir = "index_data"
index.storage_context.persist(index_dir)
print(f"Index saved to {index_dir}")

# 示例查询
# 从存储加载索引
storage_context = StorageContext.from_defaults(persist_dir=index_dir)
loaded_index = load_index_from_storage(storage_context, embed_model=embedding_model)

# 创建查询引擎
query_engine = loaded_index.as_query_engine(llm=llm)

# 执行示例查询
query_text = "公司总共赔了小杨多少钱？"
response = query_engine.query(query_text)
print("\nQuery Results:")
print(f"Question: {query_text}")
print(f"Answer: {response}")
