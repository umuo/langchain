# --coding: utf-8--

# 这里支持 Ollama, OpenAI, AzureOpenAI, VLLM, Xinference
from langchain_ollama import OllamaLLM
from langchain.callbacks import StdOutCallbackHandler
from langchain.schema.runnable import RunnableConfig
from llama_index.embeddings.ollama import OllamaEmbedding
from llama_index.llms.ollama import Ollama
import httpx
from dotenv import load_dotenv
import os
from autogen_ext.models.openai import OpenAIChatCompletionClient
load_dotenv(os.path.join(os.path.dirname(os.path.dirname(__file__)), '.env'))
load_dotenv(os.path.join(os.path.dirname(os.path.dirname(__file__)), '.env.local'), override=True)

OLLAMA_CHAT_MODEL = os.getenv("OLLAMA_CHAT_MODEL")
OLLAMA_URL = os.getenv("OLLAMA_URL")
OLLAMA_EMEMBEDDING_MODEL = os.getenv("OLLAMA_EMEMBEDDING_MODEL")

print(OLLAMA_EMEMBEDDING_MODEL, OLLAMA_URL, OLLAMA_CHAT_MODEL)

ollama_embedding = OllamaEmbedding(
    model_name=OLLAMA_EMEMBEDDING_MODEL,  # 指定使用的 Ollama 模型名称
    base_url=OLLAMA_URL,  # 指定 Ollama 服务的 URL
    ollama_additional_kwargs={"timeout": 120.0}  # 设置超时时间为 120 秒
)

llama_index_ollama = Ollama(
    model=OLLAMA_CHAT_MODEL,  # 指定使用的 Ollama 模型名称
    base_url=OLLAMA_URL,  # 指定 Ollama 服务的 URL
    client=httpx.Client(timeout=60),
)

llm = OllamaLLM(
    model=OLLAMA_CHAT_MODEL,
    base_url=OLLAMA_URL,
    temperature=0.7,  # 控制输出的随机性
    max_tokens=150,  # 限制输出的最大标记数
    top_p=0.9,  # 用于核采样的概率阈值
    frequency_penalty=0.5,  # 降低频繁使用的词的概率
    presence_penalty=0.5,  # 鼓励模型使用新的词
)

autogen_llm = OpenAIChatCompletionClient(
    api_key=os.getenv("OPENAI_API_KEY"),
    base_url=os.getenv("OPENAI_BASE_URL"),
    model=os.getenv("CHAT_MODEL_NAME"),
    model_info={
        "json_output": False,
        "vision": False,
        "function_calling": True,
        "family": "unknown",
        "structured_output": False,
    }
)

# response = llm.invoke("你是谁")
# print(response)

# 创建一个简单的回调处理器
# callback_handler = StdOutCallbackHandler()
# 创建一个RunnableConfig对象
# config = RunnableConfig(
#     callbacks=[callback_handler],
#     tags=["test"],  # 设置回调处理器的标签
#     metadata={"session_id": "123"},  # 设置回调处理器的元数据
# )
#
# for chunk in llm.stream(input="你是谁",
#                         config=config):
#     print(chunk, end="", flush=True)
