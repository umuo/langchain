#!/usr/bin/python3
# --*-- coding: utf-8 --*--
# @Author: gitsilence
# @Time: 2025/5/17 17:19
from dataclasses import dataclass
from typing import Callable
from autogen_core import DefaultTopicId, MessageContext, RoutedAgent, default_subscription, message_handler
from autogen_core import AgentId, SingleThreadedAgentRuntime
import asyncio


@dataclass  # 该装饰器用于自动生成类的构造函数，__init__()、__repr__()、__eq__()
class Message:
    content: int


@default_subscription
class Modifier(RoutedAgent):
    def __init__(self, modify_val: Callable[[int], int]) -> None:
        super().__init__("A modifier agent.")
        self.modify_val = modify_val

    @message_handler
    async def handle_message(self, message: Message, ctx: MessageContext) -> None:
        val = self.modify_val(message.content)
        print(f"{'-'*80}\nModifier:\nModified {message.content} to {val}")
        await self.publish_message(Message(content=val), DefaultTopicId())


@default_subscription
class Checker(RoutedAgent):
    def __init__(self, run_until: Callable[[int], bool]) -> None:
        super().__init__("A checker agent.")
        self.run_until = run_until

    @message_handler
    async def handle_message(self, message: Message, ctx: MessageContext) -> None:
        if not self.run_until(message.content):
            print(f"{'-' * 80}\nChecker:\n{message.content} passed the check, continue.")
            await self.publish_message(Message(content=message.content), DefaultTopicId())
        else:
            print(f"{'-' * 80}\nChecker:\n{message.content} failed the check, stopping.")

@default_subscription
class Logger(RoutedAgent):
    @message_handler
    async def handle_message(self, message: Message, ctx: MessageContext) -> None:
        print(f"{'-' * 80}\nLogger:\nReceived {message.content}")


async def main() -> None:
    runtime = SingleThreadedAgentRuntime()
    await Modifier.register(runtime, "modifier", lambda: Modifier(modify_val=lambda x: x - 1))
    await Checker.register(runtime, "checker", lambda: Checker(run_until=lambda x: x <= 1))
    await Logger.register(runtime, "logger", lambda: Logger(description="print log messages"))
    runtime.start()
    await runtime.send_message(Message(content=10), AgentId("checker", "default"))
    await runtime.stop_when_idle()

if __name__ == '__main__':
    asyncio.run(main())
