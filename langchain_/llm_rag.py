import streamlit as st
from langchain_community.document_loaders import PyPDFLoader, TextLoader
from langchain.text_splitter import CharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_openai import ChatOpenAI
from langchain_openai import OpenAIEmbeddings
from langchain.embeddings.base import Embeddings
from typing import List, Dict
import os
import traceback
import hashlib

from langchain.memory import ConversationBufferMemory

# 设置页面配置
st.set_page_config(page_title="知识库聊天助手", layout="wide")

# 设置 OpenAI API key
os.environ["OPENAI_API_KEY"] = "sk-xxxx"
os.environ["OPENAI_API_BASE"] = "https://newapi.xxx.com/v1"  # 添加自定义 base URL

# 模型配置
EMBEDDING_MODEL = "shaw/dmeta-embedding-zh"  # 例如: "text-embedding-3-small"
CHAT_MODEL = "gemini-exp-1206"  # 例如: "gpt-3.5-turbo"

# 初始化向量存储
def initialize_vectorstore():
    if 'vectorstore' not in st.session_state:
        st.session_state.vectorstore = None

# 初始化聊天历史
def initialize_chat_history():
    if 'chat_history' not in st.session_state:
        st.session_state.chat_history = []

# 处理文件上传
def process_file(uploaded_file):
    # 获取文件扩展名
    file_extension = uploaded_file.name.split(".")[-1]
    
    # 保存上传的文件
    with open(f"temp.{file_extension}", "wb") as f:
        f.write(uploaded_file.getbuffer())
    
    try:
        # 根据文件类型选择加载器
        if file_extension == "pdf":
            loader = PyPDFLoader(f"temp.{file_extension}")
        else:
            loader = TextLoader(f"temp.{file_extension}")
        
        # 加载文档
        documents = loader.load()
        
        # 修改文本分割器的参数
        text_splitter = CharacterTextSplitter(
            separator="\n",
            chunk_size=500,  # 减小chunk大小
            chunk_overlap=50,  # 减小overlap
            length_function=len,
            is_separator_regex=False,
        )
        texts = text_splitter.split_documents(documents)
        
        # 确保文本不为空
        if not texts:
            st.error("文档内容为空，请检查文件内容")
            return None
            
        # 使用CustomEmbeddings
        embeddings = CustomEmbeddings(
            model=EMBEDDING_MODEL,
            api_base=os.getenv("OPENAI_API_BASE")
        )
        
        # 打印调试信息
        print(f"准备处理的文本数量: {len(texts)}")
        
        # 详细的API响应调试
        try:
            test_text = texts[0].page_content[:100]
            print(f"测试文本: {test_text}")
            
            # 测试嵌入
            test_embedding = embeddings.embed_query(test_text)
            print(f"测试嵌入向量维度: {len(test_embedding)}")
            print(f"嵌入向量前10个值: {test_embedding[:10]}")
            
        except Exception as e:
            print(f"测试嵌入失败: {str(e)}")
            print(f"错误类型: {type(e)}")
            raise
        
        # 添加错误处理和重试逻辑
        max_retries = 3
        for attempt in range(max_retries):
            try:
                vectorstore = FAISS.from_documents(
                    documents=texts,
                    embedding=embeddings
                )
                break
            except Exception as e:
                print(f"尝试 {attempt + 1}/{max_retries} 失败: {str(e)}")
                if attempt == max_retries - 1:
                    raise
                continue
        
        return vectorstore
        
    except Exception as e:
        st.error(f"处理文件时出错: {str(e)}")
        traceback.print_exc()  # 打印完整的错误堆栈
        return None
    finally:
        # 删除临时文件
        if os.path.exists(f"temp.{file_extension}"):
            os.remove(f"temp.{file_extension}")

class CustomEmbeddings(Embeddings):
    def __init__(self, model: str, api_base: str):
        self.model = model
        self.api_base = api_base
        self.client = OpenAIEmbeddings(api_key=os.getenv("OPENAI_API_KEY"), base_url=api_base)
    
    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        """Embed search docs."""
        try:
            # 确保每个文本都单独处理
            embeddings = []
            for text in texts:
                response = self.client.client.create(
                    model=self.model,
                    input=[text]  # 注意这里需要传入列表
                )
                embedding = response.data[0].embedding
                if len(embedding) == 0:
                    raise ValueError(f"检测到空的嵌入向量，文本: {text[:100]}...")
                embeddings.append(embedding)
            
            print(f"成功生成嵌入向量数量: {len(embeddings)}")
            return embeddings
            
        except Exception as e:
            print(f"嵌入文档失败: {str(e)}")
            raise
    
    def embed_query(self, text: str) -> List[float]:
        """Embed query text."""
        try:
            response = self.client.client.create(
                model=self.model,
                input=[text]
            )
            embedding = response.data[0].embedding
            if len(embedding) == 0:
                raise ValueError("检测到空的嵌入向量")
            return embedding
        except Exception as e:
            print(f"嵌入查询失败: {str(e)}")
            raise

def get_file_hash(file_content: bytes) -> str:
    """计算文件的哈希值"""
    return hashlib.md5(file_content).hexdigest()

def load_processed_files() -> Dict[str, bool]:
    """从session_state加载已处理文件的记录"""
    if 'processed_files' not in st.session_state:
        st.session_state.processed_files = {}
    return st.session_state.processed_files

def mark_file_as_processed(file_hash: str):
    """标记文件为已处理"""
    st.session_state.processed_files[file_hash] = True

def is_file_processed(file_hash: str) -> bool:
    """检查文件是否已处理"""
    return st.session_state.processed_files.get(file_hash, False)

@st.dialog("引用内容")
def show_sources(sources):
    if not sources:
        st.warning("没有找到引用内容")
        return
        
    for i, source in enumerate(sources, 1):
        st.markdown(f"### 引用 {i}")
        st.markdown(f"```\n{source['content']}\n```")
        if source['source'] != '未知来源':
            st.caption(f"来源: {source['source']}")
        st.markdown("---")

def get_answer(question: str, vectorstore, chat_history=None):
    try:
        llm = ChatOpenAI(
            model=CHAT_MODEL,
            temperature=0.7,
            base_url=os.getenv("OPENAI_API_BASE"),
            streaming=True
        )
        
        # 检索相关文档并保存
        docs = vectorstore.similarity_search(question, k=3)
        sources = [{"content": doc.page_content, "source": getattr(doc.metadata, 'source', '未知来源')} 
                  for doc in docs]
        
        # 构建提示模板
        template = """基于以下已知信息，简洁和专业的来回答用户的问题。
        如果无法从中得到答案，请说 "抱歉，我无法从已知信息中找到答案。"
        已知信息:
        {context}
        
        聊天历史：
        {chat_history}
        
        问题: {question}
        回答: """
        
        # 格式化聊天历史
        chat_history_str = ""
        if chat_history:
            chat_history_str = "\n".join([f"问：{q}\n答：{a}" if a else f"问：{q}" 
                                         for q, a in chat_history])
        
        # 构建提示
        prompt = template.format(
            context="\n".join([doc.page_content for doc in docs]),
            chat_history=chat_history_str,
            question=question
        )
        with st.chat_message("assistant"):
            # 流式输出回答
            response_placeholder = st.empty()
            full_response = ""
            
            for chunk in llm.stream(prompt):
                content = chunk.content
                full_response += content
                response_placeholder.markdown(full_response + "▌")
            
            response_placeholder.markdown(full_response)
        
            # 检查回答是否来自知识库
            if not "抱歉，我无法从已知信息中找到答案" in full_response:
                # 添加引用来源链接
                if st.button("📚 引用来源", key=f"source_btn_{hash(question)}", type="secondary", use_container_width=False):
                    show_sources(sources)
        
        return full_response, sources
        
    except Exception as e:
        st.error(f"获取回答时出错: {str(e)}")
        return "抱歉，生成回答时出现错误。", []

def main():
    st.title("文档问答系统")
    
    # 初始化session state
    if 'uploaded_files' not in st.session_state:
        st.session_state.uploaded_files = []
    if 'vectorstore' not in st.session_state:
        st.session_state.vectorstore = None
    if 'processed_files' not in st.session_state:
        st.session_state.processed_files = {}
    if 'chat_history' not in st.session_state:
        st.session_state.chat_history = []
    
    # 创建左右布局
    left_col, right_col = st.columns([1, 2])
    
    # 左侧：文件上传区域
    with left_col:
        st.header("文档上传")
        uploaded_files = st.file_uploader("上传文档", 
                                        type=["txt", "md", "pdf"], 
                                        accept_multiple_files=True)
        
        # 更新上传文件列表
        if uploaded_files:
            st.session_state.uploaded_files = uploaded_files
        
        # 显示已上传文件列表
        if st.session_state.uploaded_files:
            st.write("已上传文件：")
            for file in st.session_state.uploaded_files:
                file_hash = get_file_hash(file.getvalue())
                status = "已处理" if st.session_state.processed_files.get(file_hash, False) else "未处理"
                st.write(f"- {file.name} ({status})")
        
        # 提交按钮
        if st.button("开始处理文件"):
            if not st.session_state.uploaded_files:
                st.warning("请先上传文件")
            else:
                with st.spinner("正在处理文件..."):
                    for file in st.session_state.uploaded_files:
                        file_hash = get_file_hash(file.getvalue())
                        if not st.session_state.processed_files.get(file_hash, False):
                            st.write(f"处理文件: {file.name}")
                            vectorstore = process_file(file)
                            if vectorstore:
                                if st.session_state.vectorstore is None:
                                    st.session_state.vectorstore = vectorstore
                                else:
                                    st.session_state.vectorstore.merge_from(vectorstore)
                                st.session_state.processed_files[file_hash] = True
                                st.success(f"{file.name} 处理完成")
                            else:
                                st.error(f"{file.name} 处理失败")
                        else:
                            st.info(f"跳过已处理的文件: {file.name}")
    
    # 右侧：聊天界面
    with right_col:
        st.header("对话")
        
        with st.container(border=True):
            # 显示聊天历史（正序显示所有消息）
            if st.session_state.vectorstore is not None:
                for i, message in enumerate(st.session_state.chat_history):
                    with st.chat_message(message["role"]):
                        st.write(message["content"])
                        if (message["role"] == "assistant" and 
                            "抱歉，我无法从已知信息中找到答案" not in message["content"] and
                            message.get("sources")):
                            if st.button("📚 引用来源", key=f"source_btn_{i}", type="secondary", use_container_width=False):
                                show_sources(message["sources"])
            
            question = st.chat_input("请输入您的问题")
            
            if question:
                # 显示用户问题
                with st.chat_message("user"):
                    st.write(question)
                
                # 添加用户问题到历史记录
                st.session_state.chat_history.append({
                    "role": "user",
                    "content": question
                })
                
                with st.spinner("正在生成回答..."):                
                    # 生成回答
                    response, sources = get_answer(
                        question, 
                        st.session_state.vectorstore,
                        chat_history=[(msg["content"], None) for msg in st.session_state.chat_history[:-1]]
                    )
                    
                    # 添加助手回答到历史记录
                    st.session_state.chat_history.append({
                        "role": "assistant", 
                        "content": response,
                        "sources": sources
                    })
                
                # 强制重新渲染
                st.rerun()

if __name__ == "__main__":
    main()
