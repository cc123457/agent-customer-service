"""有限、可解释的事实规范化：预算同属性键与明确ISO生效时间，不猜测任意语义冲突。"""
import hashlib
import re
from dataclasses import replace
from datetime import datetime, time, timezone, timedelta

BUSINESS_TZ = timezone(timedelta(hours=8))
EFFECTIVE_DATE = re.compile(r"(?:从|自)\s*(\d{4}-\d{2}-\d{2})(?:\s*起|\s*开始|\s*生效)")


def effective_timestamp(value):
    """将明确日期按UTC+8零点解释；带时间必须有时区，不支持相对时间或模糊日期。"""
    if not value:
        return None
    if re.fullmatch(r"\d{4}-\d{2}-\d{2}", value):
        return datetime.combine(datetime.strptime(value, "%Y-%m-%d").date(), time(), BUSINESS_TZ).timestamp()
    moment = datetime.fromisoformat(value.replace("Z", "+00:00"))
    if moment.tzinfo is None:
        raise ValueError("生效时间必须是ISO日期或带时区时间")
    return moment.timestamp()


def explicit_effective_date(text):
    """只提取一个明确生效日期，多日期交给更细粒度模型/澄清，非法日历值不当成即时事实。"""
    values = list(dict.fromkeys(EFFECTIVE_DATE.findall(text)))
    if len(values) > 1:
        raise ValueError("同一句包含多个生效日期，不能全局绑定")
    value = values[0] if values else ""
    effective_timestamp(value)
    return value


def note_key(value):
    """有限识别预算属性：配件/设备/一般采购分开，其余笔记继续按内容哈希，避免误合并。"""
    if re.search(r"预算(?:上限|最多|不超过|限额)?[^。\n]{0,20}(?:元|块|人民币)", value):
        subject = "accessories" if "配件" in value else "device" if re.search(r"购机|买机器|设备预算|机器人预算", value) else "purchase"
        return "note:budget:" + subject
    return "note:" + hashlib.sha256(value.encode("utf-8")).hexdigest()[:24]


def normalize_candidate(candidate, source):
    """同一规范化同时用于规则、模型及SQL入口，生效时间和对象都必须能追溯原文。"""
    key = note_key(candidate.value) if candidate.key.startswith("note:") and candidate.action != "delete" else candidate.key
    value = candidate.valid_at or explicit_effective_date(source)
    if value and value not in source:
        raise ValueError("生效时间缺少原文依据")
    effective_timestamp(value)
    return replace(candidate, key=key, valid_at=value)


def current_rows(rows):
    """读旧库时对同对象预算取最新来源，兼容旧哈希笔记；不删除原文或擅改其他自由笔记。"""
    selected = {}
    for row in rows:
        attribute = row.get("attribute", row["memory_key"])
        canonical = note_key(row["value"]) if attribute.startswith("note:") else attribute
        identity = (row.get("entity_id", "profile"), canonical)
        if identity not in selected or row["source_message_id"] > selected[identity]["source_message_id"]:
            selected[identity] = row
    return sorted(selected.values(), key=lambda row: row["source_message_id"], reverse=True)
