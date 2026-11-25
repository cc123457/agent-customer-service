"""移除模型JSON外层代码围栏，不提取任意正文中的JSON，也不改字段、类型或业务值。"""
import re


def json_payload(text):
    """兼容完整围栏及末尾孤立围栏；其余正文保留给严格解析器拒绝，原始输出由调用方留存。"""
    if not isinstance(text, str):
        raise TypeError("模型JSON输出必须是文本")
    value = text.strip()
    value = re.sub(r"\A```(?:json)?[ \t]*\r?\n", "", value, count=1, flags=re.IGNORECASE)
    value = re.sub(r"\r?\n```[ \t]*\Z", "", value, count=1)
    return value.strip()
