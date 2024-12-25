import streamlit as st
import requests
import time
import json

# 页面配置
st.set_page_config(page_title="LLM Chat", page_icon="🤖", layout="wide")

st.title("🤖 LLM Chat Interface")

# --- 侧边栏设置 ---
with st.sidebar:
    st.header("🔧 配置")
    api_url = st.text_input("API URL", value="https://newapi.lacknb.com/v1/chat/completions")
    api_key = st.text_input("API Key", type="password")
    model = st.selectbox("选择模型", ["gpt-3.5-turbo", "gpt-4", "grok-beta"])
    temperature = st.slider("Temperature", 0.0, 1.0, 0.7)
    max_tokens = st.number_input("Max Tokens", min_value=50, max_value=4000, value=1000)

# --- 聊天记录 ---
if "messages" not in st.session_state:
    st.session_state.messages = []

if "input_disabled" not in st.session_state:
    st.session_state.input_disabled = False

# 展示历史聊天记录
for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])

# --- 用户输入 ---
user_input = st.chat_input("输入消息...", disabled=st.session_state.input_disabled)

if user_input:
    # 禁用输入框
    st.session_state.input_disabled = True
    # 显示用户输入
    st.session_state.messages.append({"role": "user", "content": user_input})
    with st.chat_message("user"):
        st.markdown(user_input)

    # --- 调用 API ---
    headers = {
        "Content-Type": "application/json",
        "Authorization": f"Bearer {api_key}"
    }
    payload = {
        "model": model,
        "messages": st.session_state.messages,
        "temperature": temperature,
        "max_tokens": max_tokens,
        "stream": True
    }

    with st.chat_message("assistant"):
        message_placeholder = st.empty()
        response = requests.post(api_url, json=payload, headers=headers, stream=True)
        if response.status_code == 200:
            message = ""
            # print(response.text)
            # 逐字输出
            for chunk in response.iter_lines(decode_unicode=True):
                # print(chunk)
                if chunk.startswith('data:'):
                    data = chunk[5:]  # 去掉'data: '前缀
                    if data.strip() != '[DONE]':  # 忽略结束标志
                        data = json.loads(data)
                        content = data.get("choices", [{}])[0].get("delta", {}).get("content", "")
                        message += content
                        message_placeholder.markdown(message + "▌")  # 更新同一行文本
                        time.sleep(0.05)
            # 记录AI回复
            st.session_state.messages.append({"role": "assistant", "content": message})
        else:
            st.error(f"API请求失败: {response.status_code}")
            st.write(response.text)

# --- 清空输入框 ---
# st.session_state.input_disabled = False
