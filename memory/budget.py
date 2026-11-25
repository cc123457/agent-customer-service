"""摘要与上下文共享预算估算，不反向导入聊天业务服务。"""


def estimate_tokens(text: str) -> int:
    """返回UTF-8字节数作为保守预算单位，不冒充供应商精确Token数。"""
    return len(text.encode("utf-8"))


def message_tokens(messages: list[dict]) -> int:
    """估算正文与每条固定开销，不包括系统提示词、工具Schema及Agent后续工具结果。"""
    return sum(estimate_tokens(message["content"]) + 12 for message in messages)
