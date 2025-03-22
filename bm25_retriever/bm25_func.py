#!/usr/bin/python3
# --*-- coding: utf-8 --*--
# @Author: gitsilence
# @Time: 2025/3/22 23:31
import math
from collections import Counter


class BM25:
    def __init__(self, documents, k1=1.5, b=0.75):
        self.documents = documents  # 语料库
        self.k1 = k1  # 词频饱和参数
        self.b = b  # 文档长度归一化参数
        self.doc_count = len(documents)  # 总文档数
        self.avg_doc_len = sum(len(doc.split()) for doc in documents) / self.doc_count  # 平均文档长度
        self.inverted_index = self._build_inverted_index()  # 构建倒排索引
        self.idf = self._compute_idf()  # 计算 IDF 值

    def _build_inverted_index(self):
        """构建倒排索引"""
        inverted_index = {}
        for doc_id, doc in enumerate(self.documents):
            words = doc.split()
            word_freq = Counter(words)  # 计算文档内词频
            for word, freq in word_freq.items():
                if word not in inverted_index:
                    inverted_index[word] = {}
                inverted_index[word][doc_id] = freq  # 记录词在文档中的出现次数
        return inverted_index

    def _compute_idf(self):
        """计算 IDF（逆文档频率）"""
        idf = {}
        for word, doc_freqs in self.inverted_index.items():
            df = len(doc_freqs)  # 包含该词的文档数
            idf[word] = math.log((self.doc_count - df + 0.5) / (df + 0.5) + 1)  # BM25 IDF 计算公式
        return idf

    def _bm25_score(self, query, doc_id):
        """计算单个文档的 BM25 相关性得分"""
        score = 0
        doc_words = self.documents[doc_id].split()
        doc_len = len(doc_words)  # 当前文档的长度
        for word in query.split():
            if word in self.inverted_index:
                if doc_id in self.inverted_index[word]:
                    f = self.inverted_index[word][doc_id]  # 词频
                    idf = self.idf[word]  # 词的 IDF
                    numerator = f * (self.k1 + 1)
                    denominator = f + self.k1 * (1 - self.b + self.b * (doc_len / self.avg_doc_len))
                    score += idf * (numerator / denominator)  # 计算 BM25 评分
        return score

    def search(self, query, top_k=2):
        """对所有文档计算 BM25 评分并返回 Top K 相关文档"""
        scores = [(doc_id, self._bm25_score(query, doc_id)) for doc_id in range(self.doc_count)]
        scores = sorted(scores, key=lambda x: x[1], reverse=True)  # 按照得分降序排序
        return [self.documents[doc_id] for doc_id, score in scores[:top_k]]
        # return [self.documents[doc_id] for doc_id, score in scores[:top_k] if score > 0]  # 过滤掉得分为 0 的文档


# 示例文档
documents = [
    "I like apples",
    "I like oranges",
    "Apples and oranges are fruits",
    "I like computers by Apple",
    "I love fruit juice"
]

# 创建 BM25 检索器
bm25 = BM25(documents)

# 查询 "Apple"
query = "Apple"
results = bm25.search(query)

# 输出结果
for doc in results:
    print(doc)
