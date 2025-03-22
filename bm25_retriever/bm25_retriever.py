#!/usr/bin/python3
# --*-- coding: utf-8 --*--
# @Author: gitsilence
# @Time: 2025/3/22 22:20
from langchain_community.retrievers.bm25 import BM25Retriever
"""
pip install rank_bm25
"""

doc_list = [
    "I like apples",
    "I like oranges",
    "Apples and oranges are fruits",
    "I like computers by Apple",
    "I love fruit juice"
]
bm25_retriever = BM25Retriever.from_texts(doc_list)
bm25_retriever.k = 2
result = bm25_retriever.invoke("Apple")
# print(result
for d in result:
    print(d)
# page_content='I like computers by Apple'
# page_content='I love fruit juice'
