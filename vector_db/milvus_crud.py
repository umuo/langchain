from pymilvus import (
    connections,
    utility,
    FieldSchema,
    CollectionSchema,
    DataType,
    Collection,
)
import numpy as np
import requests


def get_embedding(text):
    response = requests.post("http://10.118.21.135:11434/api/embeddings",
                             json={
                                 "model": "quentinz/bge-large-zh-v1.5:latest",
                                 "prompt": text
                             })
    return response.json()['embedding']


# 连接到 Milvus 服务器
def connect_to_milvus(host='10.118.21.135', port='19530'):
    connections.connect(host=host, port=port)


# 创建集合
def create_collection(collection_name, dim=1024):  # Changed from 4096 to 1024 for BGE model
    # 定义字段
    fields = [
        FieldSchema(name="id", dtype=DataType.INT64, is_primary=True),
        FieldSchema(name="title", dtype=DataType.VARCHAR, max_length=200),
        FieldSchema(name="category", dtype=DataType.VARCHAR, max_length=100),
        FieldSchema(name="content", dtype=DataType.VARCHAR, max_length=2000),
        FieldSchema(name="embedding", dtype=DataType.FLOAT_VECTOR, dim=dim)
    ]
    
    schema = CollectionSchema(fields=fields, description="文档向量集合")
    collection = Collection(name=collection_name, schema=schema)

    index_params = {
        "metric_type": "L2",
        "index_type": "IVF_FLAT",
        "params": {"nlist": 1024}
    }
    collection.create_index(field_name="embedding", index_params=index_params)
    return collection


# 插入数据
def insert_data(collection, titles, categories, contents, ids=None):
    if ids is None:
        ids = list(range(len(titles)))

    embeddings = [get_embedding(content) for content in contents]

    entities = [
        ids,
        titles,
        categories,
        contents,
        embeddings
    ]

    collection.insert(entities)
    collection.flush()


# 更新数据
def update_data(collection, id, title, category, content):
    new_embedding = get_embedding(content)

    collection.upsert([
        [id],
        [title],
        [category],
        [content],
        [new_embedding]
    ])
    collection.flush()


# 精确匹配查询
def exact_search(collection, field_name, value):
    collection.load()
    results = collection.query(
        expr=f"{field_name} == '{value}'",
        output_fields=["id", "title", "category", "content"]
    )
    return results


# 语义相似度搜索
def semantic_search(collection, query_text, top_k=5):
    query_vector = get_embedding(query_text)

    collection.load()
    search_params = {"metric_type": "L2", "params": {"nprobe": 10}}

    results = collection.search(
        data=[query_vector],
        anns_field="embedding",
        param=search_params,
        limit=top_k,
        output_fields=["id", "title", "category", "content"]
    )

    return results


# 使用示例
if __name__ == "__main__":
    connect_to_milvus()

    collection_name = "test_document_collection"
    if utility.has_collection(collection_name):
        utility.drop_collection(collection_name)

    collection = create_collection(collection_name)

    # 插入示例数据
    titles = ["Python教程", "机器学习入门", "深度学习基础"]
    categories = ["编程语言", "人工智能", "人工智能"]
    contents = [
        "Python是一种简单易学的编程语言",
        "机器学习是人工智能的重要分支",
        "深度学习是机器学习中的一个重要方向"
    ]
    insert_data(collection, titles, categories, contents)

    # 更新数据
    update_data(collection, 0, "Python进阶教程", "编程语言", "Python是一种强大的编程语言")

    # 精确匹配查询（按分类）
    results = exact_search(collection, "category", "人工智能")
    print("分类查询结果:", results)

    # 语义搜索
    results = semantic_search(collection, "编程入门教程", top_k=2)
    print("语义搜索结果:", results[0])

    connections.disconnect("default")
