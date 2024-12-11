from base.base_llm import llm
from langchain_core.runnables.history import RunnableWithMessageHistory
from langchain_core.chat_history import BaseChatMessageHistory
from langchain.prompts import ChatPromptTemplate, MessagesPlaceholder
from pydantic import BaseModel, Field
from langchain_core.messages import BaseMessage, AIMessage
from typing import List, Sequence

"""
带有记忆的对话助手
"""

class InMemoryHistory(BaseChatMessageHistory, BaseModel):
    """In memory implementation of chat message history."""

    messages: List[BaseMessage] = Field(default_factory=list)

    def add_messages(self, messages: Sequence[BaseMessage]) -> None:
        """Add a list of messages to the store"""
        self.messages.extend(messages)

    def clear(self) -> None:
        self.messages = []

# 根据session_id获取对话历史
store = {}

def get_by_session_id(session_id: str) -> BaseChatMessageHistory:
    if session_id not in store:
        store[session_id] = InMemoryHistory()
    return store[session_id]

# 这里还可以同时使用 user_id, conversation_id 
def get_session_history(user_id: str, conversation_id: str) -> BaseChatMessageHistory:
    if (user_id, conversation_id) not in store:
        store[(user_id, conversation_id)] = InMemoryHistory()
    return store[(user_id, conversation_id)]

prompt = ChatPromptTemplate.from_messages([
    ("system", "你是乐于助人的AI助手, 你擅长{ability}"),
    MessagesPlaceholder(variable_name="history"),
    ("human", "{input}"),
])

chain = prompt | llm
# 构建对话链
chain_with_history = RunnableWithMessageHistory(chain,
    get_by_session_id,
    input_messages_key="input",
    output_messages_key="output",
    history_messages_key="history"
)

# 测试对话助手
resp = chain_with_history.stream(input={"ability": "解答问题", "input": "你叫什么名字"}, config={"configurable": {"session_id": "foo"}})
response_text = ""
for chunk in resp:
    response_text += chunk
    print(chunk, end="", flush=True)
print()
# print(response_text)
print(chain_with_history.invoke(input={"ability": "解答问题", "input": "我上一个问题问的什么"}, config={"configurable": {"session_id": "foo"}}))
print(store)