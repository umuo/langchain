# --coding: utf-8--

import sys
import os
# Add project root to Python path
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
from base.base_llm import llama_index_ollama, ollama_embedding
from llama_index.core.query_engine import NLSQLTableQueryEngine
from database_operate_sample import sql_database, table_name

query_engine = NLSQLTableQueryEngine(
    sql_database=sql_database,
    tables=[table_name],
    llm=llama_index_ollama,
    embed_model=ollama_embedding
)

response = query_engine.query("哪个城市最受欢迎？")
print(response)
