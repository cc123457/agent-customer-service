"""所有模型上下文共用的遗忘策略；档案保留与允许模型使用是两件事。"""
import json
import re
from dataclasses import dataclass


ATTRIBUTE_WORDS = {
    "device_model": r"型号|机器人|设备|我用的|我使用的|换成|换了|新买",
    "floor_type": r"地面|地板|瓷砖|地毯",
    "pet_status": r"养宠|养猫|养狗|有猫|有狗|没有猫|没有狗",
    "reply_style": r"回答|回复|风格|简短|详细",
}


@dataclass(frozen=True)
class MemoryAccessPolicy:
    """一次SQL快照得到的本人遗忘规则与版本；不接受模型提供的用户身份。"""
    version: int = 0
    rules: tuple = ()

    def allows(self, message, entity_context=""):
        """旧来源命中被遗忘属性/值时整条丢弃；不会全局禁止公共型号词汇。"""
        content = message["content"]
        for rule in self.rules:
            if message["id"] > rule["cutoff_id"]:
                continue
            entity = rule["entity_id"]
            explicit_entities = {name for name in ("客厅", "卧室", "书房", "厨房", "阳台") if name in entity_context}
            if entity != "*" and explicit_entities and entity not in explicit_entities:
                continue
            if entity != "*" and entity not in content:
                # 助手常省略对象名；命中旧值时仍须排除。
                scoped = False
            else:
                scoped = True
            values = json.loads(rule["values_json"])
            literal = any(value and re.search(
                r"(?<![A-Za-z0-9_-])" + re.escape(value) + r"(?![A-Za-z0-9_-])"
                if re.fullmatch(r"[A-Za-z0-9_-]+", value) else re.escape(value), content,
            ) for value in values)
            words = ATTRIBUTE_WORDS.get(rule["attribute"], r"预算" if rule["attribute"].startswith("note:budget:") else "")
            if literal or (scoped and words and re.search(words, content)):
                return False
            # 无法从自由笔记可靠定位所有转述：保守关闭删除前的旧片段。
            if rule["attribute"].startswith("note:") and not rule["attribute"].startswith("note:budget:"):
                return False
        return True

    def messages(self, rows):
        """按整轮过滤，避免仅留下助手对被删除用户事实的转述。"""
        contexts = {}
        for row in rows:
            if row["role"] == "user":
                contexts[row["turn_id"]] = row["content"]
        blocked = {row["turn_id"] for row in rows if not self.allows(row, contexts.get(row["turn_id"], ""))}
        return [row for row in rows if row["turn_id"] not in blocked]

    def turns(self, turns):
        """历史排序前与SQL复核后都调用，日期过滤不能绕过遗忘规则。"""
        return [turn for turn in turns if len(self.messages(turn["messages"])) == len(turn["messages"])]

    def facts(self, rows):
        """对缓存/向量返回事实再按来源水位复核，阻止旧缓存重新注入。"""
        return [row for row in rows if not any(
            row["source_message_id"] <= rule["cutoff_id"]
            and row.get("attribute", row["memory_key"]) == rule["attribute"]
            and rule["entity_id"] in ("*", row.get("entity_id", "profile"))
            for rule in self.rules
        )]
