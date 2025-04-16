#!/usr/bin/python3
# --*-- coding: utf-8 --*--
# @Author: gitsilence
# @Time: 2025/4/16 10:14
import requests
from langchain_core.tools import tool

@tool(description="实时获取知乎热搜")
def get_zhihu_hot_topics(top: int = 10) -> list[dict]:
        """
        获取知乎热搜
        :param top: 热搜的数量
        :param type: 热搜的类型
        :return: 热搜的列表
        """
        url = f"https://www.zhihu.com/api/v3/feed/topstory/hot-lists/total?limit={top}&desktop=true"
        response = requests.get(url)
        try:
            hot_list = []
            for d in response.json()['data']:
                hot_list.append({
                    "title": d['target']['title'],
                    "url": d['target']['url']
                })
            return hot_list
        except:
            pass
        return []


@tool(description="实时获取新浪热搜")
def get_sina_hot_topics(top: int = 10) -> list[dict]:
    """
    获取新浪微博热搜
    :param top: 热搜的数量
    :param type: 热搜的类型
    :return: 热搜的列表
    """
    url = "https://newsapp.sina.cn/api/hotlist?newsId=HB-1-snhs%2Ftop_news_list-all"
    response = requests.get(url)
    try:
        host_list = []
        data = response.json()['data']['hotList']
        for d in data:
            url = d['base']['base']['url']
            title = d['base']['dynamicName']
            host_list.append({
                "title": title,
                "url": url
            })
        return host_list[:top]
    except Exception as e:
        # return f"failed to get hot topics: {e}"
        return []


