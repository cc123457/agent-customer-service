"""上下文前的确定性冲突处理：区分已知失效值与待澄清信息，不让旧摘要直接覆盖当前事实。"""
import re


def sanitize_reference(text, superseded):
    """只遮蔽已知旧值的字面出现，不声称能理解所有自然语言矛盾；数据库原文始终不变。"""
    result = text
    for value in superseded:
        pattern = re.escape(value)
        if value.isascii():
            pattern = r"(?<![A-Za-z0-9_-])" + pattern + r"(?![A-Za-z0-9_-])"
        result = re.sub(pattern, "[该历史值已更正或停用]", result)
    return result
