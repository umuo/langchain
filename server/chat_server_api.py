import os
import contextlib
from typing import TypedDict
from collections.abc import Generator
from collections.abc import AsyncGenerator

import uvicorn
from openai import OpenAI
from openai import Stream
from fastapi import Body
from fastapi import FastAPI
from fastapi import Request
from fastapi.responses import Response
from fastapi.responses import StreamingResponse
from openai.types.chat import ChatCompletionChunk
from fastapi.middleware.cors import CORSMiddleware
from dotenv import load_dotenv
from pydantic import BaseModel, Field
from textwrap import dedent
from enum import Enum

load_dotenv("../.env.local")


class LifespanState(TypedDict):
    openai_client: OpenAI


class ChatBelong(str, Enum):
    OWN = "own"
    GROUP = "group"
    MEMBER = "member"


class ChatRequest(BaseModel):
    model: str = Field(default=os.getenv("CHAT_MODEL_NAME"), description="模型名称")
    belong: ChatBelong = Field(default=ChatBelong.GROUP, description="对话归属")
    group_account_no: str = Field(description="群聊账号", min_length=5, title="群聊账号")
    messages: list = Field(default=[], description="对话历史")
    input: str = Field(default="", description="用户输入的文本")
    stream: bool = Field(default=True, description="是否使用流式响应")
    # 后续可以方便地添加其他参数，例如：
    # temperature: float = Field(default=0.7, description="温度参数", ge=0, le=2)
    # max_tokens: int = Field(default=None, description="最大token数")


@contextlib.asynccontextmanager
async def lifespan(_: FastAPI) -> AsyncGenerator[LifespanState, None]:
    openai_client = OpenAI(
        api_key=os.getenv("OPENAI_API_KEY"),
        base_url=os.getenv("OPENAI_BASE_URL"),
    )
    yield {
        "openai_client": openai_client,
    }


app = FastAPI(lifespan=lifespan)

# 允许所有的来源访问，允许所有的方法（GET, POST, PUT, DELETE 等），允许所有的头部
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


def get_chat_record():
    with open(r"E:\DeskTop\chat.txt", "r", encoding='UTF-8') as f:
        return f.read()


def generate(client: OpenAI, chat_request: ChatRequest) -> Generator[str, None, None]:
    # 根据群聊账号获取聊天内容列表
    chat_record = get_chat_record()

    messages = [
        {"role": "system", "content": dedent(f"""
        你是一名AI办案分析专家，具备刑侦学、犯罪心理学和电子取证的专业知识。
        聊天内容在context中，context内容如下：
        <context>
        {chat_record}
        </context>
        结合聊天记录回答用户的问题。
        """)}
    ]
    if chat_request.messages:
        messages.extend(chat_request.messages)
    elif chat_request.input:
        messages.append({"role": "user", "content": chat_request.input})
    else:
        raise ValueError("问题不能为空！")
    stream: Stream[ChatCompletionChunk] = client.chat.completions.create(
        model=chat_request.model,
        messages=messages,
        stream=chat_request.stream,
    )
    if chat_request.stream:
        for event in stream:
            # 直接将原始事件数据转为JSON字符串返回
            yield f"data: {event.model_dump_json()}\n\n"
            # current_response = event.choices[0].delta.content
            # if current_response:
            #     yield current_response
    else:
        yield f"data: {stream.model_dump_json()}\n\n"


# async def generate(client: AsyncOpenAI, text: str) -> AsyncGenerator[str, None]:
#     stream: AsyncStream[ChatCompletionChunk] = await client.chat.completions.create(
#         model="yi-medium",
#         messages=[{"role": "user", "content": text}],
#         stream=True,
#     )
#     async for event in stream:
#         yield f"data: {event.model_dump_json()}\n\n"

@app.post("/chat")
async def chat(
        request: Request,
        chat_request: ChatRequest,
) -> Response:

    if chat_request.stream:
        response_gen = generate(request.state.openai_client, chat_request)
        return StreamingResponse(
            response_gen,
            media_type="text/event-stream",
            headers={
                "Cache-Control": "no-cache",
                "Connection": "keep-alive",
                "X-Accel-Buffering": "no",
            },
        )
    else:
        response_gen = generate(request.state.openai_client, chat_request)
        response_data = next(response_gen).replace("data: ", "", 1)
        return Response(content=response_data, media_type="application/json")


if __name__ == "__main__":
    uvicorn.run(app=app, host="0.0.0.0", port=8000)
