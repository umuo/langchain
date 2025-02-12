import logging
logging.basicConfig(
    level=logging.DEBUG,  # 设置为 DEBUG 级别以显示更详细的信息
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)

from langchain_community.document_loaders import PyPDFLoader

pdf_path = "/Users/gitsilence/Downloads/084557-01.pdf"

loader = PyPDFLoader(pdf_path)
# 文档加载
documents = loader.load()
# print(documents)
import re

def clean_text(text: str) -> str:
    # 删除 HTML 标签
    text = re.sub(r"<[^>]+>", "", text)
    # 删除控制字符（如 \x00-\x1F）
    text = re.sub(r"[\x00-\x1F\x7F]", "", text)
    # 其他自定义清理逻辑（如过滤 emoji）
    return text.strip()

def clean_text(text):
    text = re.sub(r'[\x00-\x1f\x7f-\x9f]', '', text)  # 删除控制字符
    text = re.sub(r'\s+', ' ', text).strip()          # 清理多余空白符
    return text
# 文本分割
from langchain.text_splitter import RecursiveCharacterTextSplitter
text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=1000,  # 每个块的大小
    chunk_overlap=100,  # 块之间的重叠部分
    separators=["\n", "。", "\n\n", "；"],
    length_function=len,
    is_separator_regex=False,
    keep_separator=True
)

splits = text_splitter.split_documents(documents)
# splits = [text for text in splits if text.strip()]  # 过滤空块


print(len(splits))
# for i, split in enumerate(splits):
#     print(f"第{i+1}个块: {split.page_content[:100]}")
#     print("-"*100)
#     if i == 2:  
#         break

from langchain_community.vectorstores import Chroma
# Chroma 是一个向量数据库，类似于FAISS，都是用来存储和检索向量嵌入的数据库，支持持久化
# 向量化 embedding
from langchain_openai import OpenAIEmbeddings
# model_name = "shaw/dmeta-embedding-zh"
model_name = "quentinz/bge-large-zh-v1.5:latest"
api_key = "sk-xxx"
api_base = "https://newapi.xxx.com/v1"

embedding_model = OpenAIEmbeddings(
    model=model_name,
    api_key=api_key,
    base_url=api_base,
    chunk_size=1
)

# 3. 手动生成嵌入并捕获错误
valid_docs = []
valid_embeddings = []

for i, doc in enumerate(splits):
    try:
        embedding = embedding_model.embed_documents([doc.page_content])[0]  # 取出嵌入向量
        if embedding and len(embedding) > 0:
            valid_docs.append(doc)
            valid_embeddings.append(embedding)
        else:
            print(f"Empty embedding for doc {i}: {doc.page_content[:50]}")
    except Exception as e:
        print(f"Embedding failed for doc {i}: {e} - Content: {doc.page_content[:50]}")
# 向量化
try:
    vectorstore = Chroma.from_documents(
        documents=valid_docs,
        embedding=embedding_model,
        persist_directory="chroma_db"  # 直接在创建时指定保存目录
    )
    print("向量化完成并保存")
except Exception as e:
    print(f"向量化过程出错: {e}")
    raise

# 加载向量化结果
vectorstore = Chroma(
    persist_directory="chroma_db",
    embedding_function=embedding_model
)

# 测试向量化结果
query = "儿童编程需要学什么?"
docs = vectorstore.similarity_search(query, k=3)

print(docs)