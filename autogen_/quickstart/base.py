#!/usr/bin/python3
# --*-- coding: utf-8 --*--
# @Author: gitsilence
# @Time: 2025/3/20 13:46
from dotenv import load_dotenv
import os

load_dotenv("../../.env.local")
print(os.getenv("OPENAI_BASE_URL"))