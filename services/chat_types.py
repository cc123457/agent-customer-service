"""共享聊天资料、结果状态与异常，不导入模型、数据库或HTTP模块。"""
from dataclasses import dataclass
from enum import StrEnum


class CompressionStatus(StrEnum):
    """压缩结果：未触发、成功、失败、提交冲突、失败冷却；供调用方记录与判断。"""
    SKIPPED = "skipped"
    SUCCESS = "success"
    FAILED = "failed"
    CONFLICT = "conflict"
    COOLDOWN = "cooldown"


class ContextBudgetError(ValueError):
    """本轮模型输入或输出超出配置预算。"""


class SessionBusyError(ValueError):
    """同一用户会话已有请求正在处理。"""


class SessionCapacityError(ValueError):
    """持久正文或记忆条数达到配额，拒绝新增且不静默删历史。"""


class SessionLeaseError(ValueError):
    """会话租约已过期或owner不匹配，旧请求无权写回。"""


@dataclass
class ChatTurn:
    """一轮请求资料；scope隔离会话，owner标识处理权，turn_id定位图检查点。"""
    scope: str
    owner: str
    messages: list[dict]
    deadline: float
    compression: CompressionStatus
    user_id: str
    session_id: str
    request_id: str
    turn_id: str
    user_message_id: int
    resume: bool = False
    direct_answer: str = ""
    memory_analysis: dict | None = None
