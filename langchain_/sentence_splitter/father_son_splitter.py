# --coding: utf-8--
from langchain.retrievers import ParentDocumentRetriever
from langchain.storage import InMemoryStore
from langchain_chroma import Chroma
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_experimental.text_splitter import SemanticChunker
# from langchain_openai import OpenAIEmbeddings
from langchain_ollama import OllamaEmbeddings
from langchain_community.document_loaders import UnstructuredMarkdownLoader, Docx2txtLoader

from dotenv import load_dotenv
import os
import re
import tiktoken

load_dotenv("../../.env.local")
"""
dify的逻辑：
1、首先将文档根据 '\n\n' 进行split
2、然后判断当前tokens是否大于chun_size，如果大于继续后面的操作，否则就pass，使用当前chunk
3、["\n\n", "。", ". ", " ", ""]，首先判断第一个字符串，在原文中是否存在，如果存在就进行split，否则继续判断第二个字符串，以此类推
4、split之后，分别计算tokens，如果小于chunk_size，则先判断当前chunk的token+下一个chunk的token是否大于chunk_size，如果大于忽略，否则将进行合并chunk，
5、让单个chunk块的token 为小于等于chunk_size的最大值
6、以上为 父文档的splitter
7、遍历每个父文档，分别再进行splitter
"""

def computer_token(text: str, model: str = "gpt2") -> int:
    """
    计算文本的token数量
    """
    try:
        encoding = tiktoken.encoding_for_model(model)
    except KeyError:
        print(f"Warning: Model '{model}' not directly supported by tiktoken. Using 'cl100k_base' as a fallback.")
        encoding = tiktoken.get_encoding("cl100k_base")
    # print(encoding)
    tokens = encoding.encode(text)
    decoded_tokens = [encoding.decode_single_token_bytes(token) for token in tokens]
    print(f"Total tokens: {len(tokens)}")
    return len(tokens)

# 或者在加载文档时预处理
def preprocess_document(document):
    document.page_content = re.sub(r'\s+', '\n', document.page_content).strip()
    return document


# 父块分割器
# default seperator 为 ["\n\n", "。", ". ", " ", ""]
parent_splitter = RecursiveCharacterTextSplitter(
    chunk_size=500,  # 每个分段的最大长度 即 字符数
    separators=["\n\n", "。"],
    add_start_index=True,
    # 自定义长度函数，可以在这里进行预处理
    # length_function=lambda text: len(re.sub(r'\s+', ' ', text))
)

# 子块分割器
child_spitter = RecursiveCharacterTextSplitter(
    chunk_size=200,
    # chunk_overlap=50,
    separators=["\n"],
    add_start_index=True,
    # 自定义长度函数，可以在这里进行预处理
    # length_function=lambda text: len(re.sub(r'\s+', ' ', text))
)

ollama_embedding = OllamaEmbeddings(
    model=os.getenv("EMBEDDING_MODEL_NAME"),
    base_url="http://192.168.210:11434",
)

child_spitter_semantic = SemanticChunker(
    ollama_embedding,
    # 可选参数：控制分块的粒度
    breakpoint_threshold_type="percentile",  # 或 "standard_deviation"
    breakpoint_threshold_amount=0.5  # 调整阈值来控制分块大小
)

# openai_embedding = OpenAIEmbeddings(
#         openai_api_key=os.getenv("OPENAI_API_KEY"),
#         openai_api_base=os.getenv("OPENAI_BASE_URL"),
#         model=os.getenv("EMBEDDING_MODEL_NAME"),
# )

# 向量存储器（用于存储子块）
vectorstore = Chroma(
    collection_name="nezha2",
    embedding_function=ollama_embedding,
    persist_directory="./temp"
)

# 文档存储（用于存储父块）
store = InMemoryStore()

# 创建父子检索器
# retriever = ParentDocumentRetriever(
#     vectorstore=vectorstore,
#     child_splitter=child_spitter,
#     parent_splitter=parent_splitter,
#     docstore=store
#     # score_threshold=0.7,
# )

# loader = UnstructuredMarkdownLoader(
#     r"E:\DeskTop\deep_research_test_data.md",
#     # mode="elements"
# )
loader = Docx2txtLoader(
    r"/Users/gitsilence/Documents/弘扬传统文化.docx"
)
documents = loader.load()
# print(documents)
print(f"加载了 {len(documents)} 个文档")

# retriever.add_documents(documents)

# 打印父文档分割结果
print("\n=== 父文档分割结果 ===")
parent_docs = parent_splitter.split_documents(documents)
print(len(parent_docs))
for i, doc in enumerate(parent_docs, 1):
    print(f"\n父文档 {i}:")
    print(f"token: {computer_token(doc.page_content)}")
    print(f"内容: {doc.page_content[:50]}...")  # 打印前50个字符
    # print(f"元数据: {doc.metadata}")
    print("----" * 20)

documents = [preprocess_document(doc) for doc in documents]
# 打印子文档分割结果
print("\n=== 子文档分割结果 ===")
child_docs = child_spitter.split_documents(documents)
print(len(child_docs))
for i, doc in enumerate(child_docs, 1):
    print(f"\n子文档 {i}:")
    print(f"内容: {doc.page_content[:50]}...")  # 打印前500个字符
    # print(f"元数据: {doc.metadata}")
    print("----" * 20)


# response = retriever.invoke("介绍下哪吒的关系网络?")
# print(response)

