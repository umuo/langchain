import weaviate
from weaviate.auth import AuthCredentials
from typing import Optional, Dict, Any
import uuid
from langchain_ollama import OllamaEmbeddings


# 创建Ollama Embeddings实例
def create_embeddings():
    return OllamaEmbeddings(
        base_url="http://10.118.21.135:11434",  # Ollama服务地址
        model="quentinz/bge-large-zh-v1.5:latest"
    )


# 创建Weaviate客户端
def create_weaviate_client():
    client = weaviate.Client(
        url="http://10.118.21.109:8001",  # Weaviate实例地址
        auth_client_secret=weaviate.auth.AuthApiKey("WVF5YThaHlkYwhGUSmCRgsX3tD5ngdN8pkih"),
        timeout_config=(5, 60),
        startup_period=None
    )
    client.batch.configure(
        # `batch_size` takes an `int` value to enable auto-batching
        # (`None` is used for manual batching)
        batch_size=100,
        # dynamically update the `batch_size` based on import speed
        dynamic=True,
        # `timeout_retries` takes an `int` value to retry on time outs
        timeout_retries=3,
    )
    return client


# 创建collection（在Weaviate中称为Class）
def create_collection(client: weaviate.Client, class_name: str = "Article"):
    # 定义class属性
    class_obj = {
        "class": class_name,
        "vectorizer": "none",  # 我们将手动提供向量
        "properties": [
            {
                "name": "title",
                "dataType": ["text"],
                "description": "文章标题",
            },
            {
                "name": "content",
                "dataType": ["text"],
                "description": "文章内容",
            },
            {
                "name": "category",
                "dataType": ["text"],
                "description": "文章分类",
            }
        ]
    }

    # 检查class是否已存在
    try:
        client.schema.get(class_name)
        print(f"Collection {class_name} 已存在")
    except weaviate.exceptions.UnexpectedStatusCodeException:
        # 创建新的class
        client.schema.create_class(class_obj)
        print(f"Collection {class_name} 创建成功")


# 添加数据（修改后的版本）
def add_article(client: weaviate.Client,
                title: str,
                content: str,
                category: str,
                embeddings: OllamaEmbeddings,
                class_name: str = "Article"):
    # 准备数据对象
    data_object = {
        "title": title,
        "content": content,
        "category": category
    }

    # 使用content生成向量
    vector = embeddings.embed_query(content)

    # 生成UUID
    data_uuid = uuid.uuid4()

    # 添加对象
    try:
        client.data_object.create(
            class_name=class_name,
            data_object=data_object,
            uuid=str(data_uuid),
            vector=vector
        )
        print(f"文章 '{title}' 添加成功")
        return str(data_uuid)
    except Exception as e:
        print(f"添加文章失败: {str(e)}")
        return None

def find_all_articles(client: weaviate.Client,
                    class_name: str = "Article"):
    try:
        result = (
            client.query
           .get(class_name, ["title", "content", "category"])
           .do()
        )
        return result["data"]["Get"][class_name]
    except Exception as e:
        print(f"查询失败: {str(e)}")
        return None

# 查询数据（修改后的版本）
def search_articles(client: weaviate.Client,
                    query: str,
                    embeddings: OllamaEmbeddings,
                    limit: int = 3,
                    class_name: str = "Article"):
    try:
        # 生成查询向量
        search_vector = embeddings.embed_query(query)

        result = (
            client.query
            .get(class_name, ["title", "content", "category"])
            .with_near_vector({"vector": search_vector})
            .with_limit(limit)
            .do()
        )

        return result["data"]["Get"][class_name]
    except Exception as e:
        print(f"查询失败: {str(e)}")
        return []

def search_articles_by_title(client: weaviate.Client,
                           title: str,
                           class_name: str = "Article"):
    try:
        result = (
            client.query
            .get(class_name, ["title", "content", "category"])
            .with_where({
                "path": ["title"],
                "operator": "Equal",
                "valueString": title
            })
            .do()
        )
        
        return result["data"]["Get"][class_name]
    except Exception as e:
        print(f"查询失败: {str(e)}")
        return []

# 更新数据
def update_article(client: weaviate.Client,
                   uuid: str,
                   properties: Dict[str, Any],
                   class_name: str = "Article"):
    try:
        client.data_object.update(
            class_name=class_name,
            uuid=uuid,
            data_object=properties
        )
        print(f"文章更新成功")
        return True
    except Exception as e:
        print(f"更新失败: {str(e)}")
        return False


# 删除数据
def delete_article(client: weaviate.Client,
                   uuid: str,
                   class_name: str = "Article"):
    try:
        client.data_object.delete(
            class_name=class_name,
            uuid=uuid
        )
        print(f"文章删除成功")
        return True
    except Exception as e:
        print(f"删除失败: {str(e)}")
        return False


if __name__ == "__main__":
    # 创建客户端连接
    client = create_weaviate_client()

    # 创建embeddings实例
    # embeddings = create_embeddings()

    # 创建collection
    # create_collection(client)

    # 添加示例文章
    # article_uuid = add_article(
    #     client,
    #     title="Java入门",
    #     content="Java入门。。。。...",
    #     category="技术,java",
    #     embeddings=embeddings
    # )

    # if article_uuid:
    #     # 搜索文章
    #     print("\n搜索结果:")
    #     results = search_articles(
    #         client,
    #         query="向量数据库",
    #         embeddings=embeddings
    #     )
    

    # 正确匹配
    # results = search_articles_by_title(client, "Weaviate入门")

    # 获取所有结果
    results = find_all_articles(client)
    for result in results:
        print(f"标题: {result['title']}")
        print(f"分类: {result['category']}")
        print(f"内容: {result['content']}")
        print("-" * 50)

        # 更新文章
        # update_success = update_article(
        #     client,
        #     article_uuid,
        #     {"title": "Weaviate入门指南（更新版）"}
        # )

        # if update_success:
        #     # 删除文章
        #     delete_success = delete_article(client, article_uuid)
        #     if delete_success:
        #         print("\n完整的CRUD操作演示完成")
