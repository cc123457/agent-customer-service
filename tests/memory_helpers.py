"""当前聊天服务的测试装配与一轮执行辅助，测试文件之间不再互相导入fixture。"""
from types import SimpleNamespace
from unittest.mock import AsyncMock

from memory.session_lease import SessionLeaseStore
from services.chat_service import ChatService


class AnswerAgent:
    async def execute_stream(self, messages, user_id):
        """只用于控制程序路径的替身，真实SQL与上下文选择不被模拟。"""
        yield {"type": "answer", "content": "测试回答"}


def service(repository, **kwargs):
    """创建独立工作租约与可控摘要器，不向真实模型发送请求。"""
    summarizer = SimpleNamespace(ainvoke=AsyncMock(return_value=SimpleNamespace(content="阶段摘要")))
    return ChatService(AnswerAgent(), SessionLeaseStore(), repository, history_window=4,
                                 summarizer=summarizer, retry_cooldown=0, **kwargs)


async def complete(chat, session, text, user="1001", request_id=None):
    """执行一轮并返回输入上下文，便于验证归档、记忆和跨会话召回。"""
    turn = await chat.open_turn(session, text, user, request_id)
    async for _ in chat.reply_turn(turn, user):
        pass
    return turn
