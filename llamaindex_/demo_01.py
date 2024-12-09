import os
from pathlib import Path
from llama_index.readers import SimpleDirectoryReader
from llama_index import VectorStoreIndex, StorageContext, load_index_from_storage


try:
    # 加载数据
    documents = SimpleDirectoryReader(data_dir).load_data()
    print(f"Loaded {len(documents)} documents from {data_dir}")

    # 构建索引
    index = VectorStoreIndex.from_documents(documents)
    print("Index built successfully")

    # 保存索引
    index_dir = "index_data"
    index.storage_context.persist(index_dir)
    print(f"Index saved to {index_dir}")

    # 示例查询
    # 从存储加载索引
    storage_context = StorageContext.from_defaults(persist_dir=index_dir)
    loaded_index = load_index_from_storage(storage_context)
    
    # 创建查询引擎
    query_engine = loaded_index.as_query_engine()
    
    # 执行示例查询
    query_text = "在文档中搜索的示例问题"
    response = query_engine.query(query_text)
    print("\nQuery Results:")
    print(f"Question: {query_text}")
    print(f"Answer: {response}")

except Exception as e:
    print(f"An error occurred: {str(e)}")
