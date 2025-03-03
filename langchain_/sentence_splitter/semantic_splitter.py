from langchain_experimental.text_splitter import SemanticChunker
from langchain_community.document_loaders import Docx2txtLoader
from langchain_ollama import OllamaEmbeddings
from dotenv import load_dotenv
import os
"""
根据语义进行分割
1、根据正则进行split，默认是按照英文的规则，如果是中文最好手动传入。例如：sentence_split_regex="(?<=[。？！])"
2、buffer_size=n 会将每n个句子合并成一个
3、对列表的句子，分别进行进行embedding
4、遍历句子列表，计算当前句子和下一个句子之间的相似度（一个距离的值）
5、number_of_chunks、breakpoint_distance_threshold ....//todo 筛选之后的chunk


distances = [1,2,3,4,5]
breakpoint_threshold_amount=30,breakpoint_threshold_amount=percentile 百分数
p = 30/100 x (5 - 1) = 1.2
由于1.2不是一个整数，所以breakpoint_threshold_amount=30对应的索引是1,2，意味着需要在2和3之间插值；
如果是一个整数，则不需要进行插值，直接返回整数索引(index)对应的值，distances[index]
30%的位置会落在这两个数之间，计算：2 + (3 - 2) * 0.2 = 2.2  其中0.2 是1.2的小数部分
distances[1]=2, distances[2]=3
使用线性插值，值=2 + 




"""

load_dotenv('../../.env.local')
print(os.getenv("EMBEDDING_MODEL_NAME"))
ollama_embedding = OllamaEmbeddings(
    model=os.getenv("EMBEDDING_MODEL_NAME"),
    base_url="http://192.168.210:11434",
)

loader = Docx2txtLoader(
    r"/Users/gitsilence/Documents/弘扬传统文化.docx"
)
documents = loader.load()
# conetnt = documents[0].page_content
# print(conetnt)

semantic_chunker = SemanticChunker(
    embeddings=ollama_embedding,
    min_chunk_size=10,  # 最小分块大小
    sentence_split_regex="(?<=[。？！])",
    breakpoint_threshold_type="percentile",
    breakpoint_threshold_amount=40,
)

semantic_chunks = semantic_chunker.split_documents(documents=documents)
print(semantic_chunks)
