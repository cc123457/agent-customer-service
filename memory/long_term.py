"""长期记忆第一版：仅从用户明确陈述提取有限字段，按身份和关键词召回。"""
import hashlib
import re
from dataclasses import dataclass
from memory.fact_rules import explicit_effective_date, normalize_candidate

import jieba


@dataclass(frozen=True)
class MemoryCandidate:
    """待写入记忆的字段、原文值和操作；不是模型自主产生的权限或业务事实。"""
    key: str
    value: str
    action: str = "upsert"
    entity: str = "profile"
    evidence: str = ""
    valid_at: str = ""
    all_entities: bool = False

    @property
    def storage_key(self):
        """同一属性按对象分开存储，保留原profile键以兼容已有记忆，避免两台设备互相覆盖。"""
        return self.key if self.entity == "profile" else "entity:" + hashlib.sha256(self.entity.encode()).hexdigest()[:20] + ":" + self.key


def extract_memories(text):
    """输入本轮用户原话，返回有限规则候选；不读取助手回答、不调用额外模型。

    支持设备型号、地面、养宠状态、回复风格和显式“记住：”笔记；
    假设场景/第三人称/禁止记忆不提取。新同键陈述替换旧值并保留来源。
    """
    try:
        explicit_effective_date(text)
    except ValueError:
        return []
    deleted = []
    for label, key in (
        ("设备型号", "device_model"), ("型号", "device_model"),
        ("地面", "floor_type"), ("养宠", "pet_status"), ("回复风格", "reply_style"),
        ("回答风格", "reply_style"),
    ):
        if re.search(rf"(?:忘记|删除)(?:我(?:的|之前的)?|之前的|以前的)?{label}", text):
            deleted.append(MemoryCandidate(key, "", "delete", all_entities=True))
    if deleted:
        return list({candidate.key: candidate for candidate in deleted}.values())
    if re.search(r"不要记住|别记住|不要保存|如果|假如|假设|朋友|别人|示例|举例|引用", text):
        return []
    entity_pattern = r"(?P<entity>客厅|卧室|书房|厨房|阳台)(?:的)?(?:扫地)?(?:机器人|设备)(?:的)?型号(?:更正为|改为|是|为|[:：])\s*(?P<value>[A-Za-z][A-Za-z0-9_-]{1,30})"
    entity_matches = list(re.finditer(entity_pattern, text))
    if entity_matches:
        grouped = {}
        for match in entity_matches:
            grouped.setdefault(match.group("entity"), []).append(match)
        return [normalize_candidate(MemoryCandidate("device_model", matches[-1].group("value"),
                                "conflict" if len({match.group("value") for match in matches}) > 1 and not re.search("更正|改为", text) else "upsert",
                                entity, text), text) for entity, matches in grouped.items()]
    patterns = {
        "device_model": [
            r"(?:我的(?:机器人|设备)?型号|实际型号|更正[：:]?型号)(?:是|为|[:：])?\s*([A-Za-z][A-Za-z0-9_-]{1,30})",
            r"我(?:用的|使用的)(?:机器人|设备)?(?:是|为|[:：])\s*([A-Za-z][A-Za-z0-9_-]{1,30})",
            r"我(?:刚|最近|已经)?(?:换了|换成了|换成|新买了)(?:一台)?\s*([A-Za-z][A-Za-z0-9_-]{1,30})",
            r"我新买的(?:扫地)?机器人(?:是|为)\s*([A-Za-z][A-Za-z0-9_-]{1,30})",
        ],
        "floor_type": [
            r"(?:我家(?:的地面)?(?:现在)?(?:全)?(?:是|用的是|铺的是)|地面(?:现在)?(?:全)?(?:是|为))\s*(木地板|瓷砖|地毯)",
        ],
        "pet_status": [
            r"(我(?:家)?(?:现在)?(?:已经)?不(?:再)?养(?:猫|狗)(?:了)?)",
            r"(我(?:家)?(?:现在)?(?:已经)?没有(?:猫|狗)(?:了)?)",
            r"(我(?:家)?(?:现在)?(?:养|有)(?:了)?(?:一只|两只|[0-9]+只)?(?:猫|狗))",
        ],
        "reply_style": [
            r"我(?:希望|喜欢|要求)[^。\n]{0,12}?回答(?:尽量)?(简短|详细)",
        ],
    }
    candidates = []
    for key, expressions in patterns.items():
        matches = [
            match for expression in expressions for match in re.finditer(expression, text)
            if not re.match(r"\s*(?:吗|么|[?？])", text[match.end():])
            and not re.search(r"[?？]|吗|么|什么|多少|是否", re.split(r"[，,。；;\n]", text[match.start():], maxsplit=1)[0])
        ]
        if matches:
            latest = max(matches, key=lambda match: match.start())
            different = {match.group(1) for match in matches}
            ambiguous = key == "device_model" and (len(different) > 1 or bool(re.search(r"也是\s*[A-Za-z][A-Za-z0-9_-]+", text)))
            action = "conflict" if ambiguous and not re.search(r"更正|实际型号|换成|改为", text) else "upsert"
            candidates.append(MemoryCandidate(key, latest.group(1), action, evidence=text))
    note = re.search(r"(?:请)?记住[：:]\s*([^\n]{1,300})", text)
    if note and not candidates:
        value = note.group(1).strip()
        if value:
            key = "note:" + hashlib.sha256(value.encode("utf-8")).hexdigest()[:24]
            candidates.append(MemoryCandidate(key, value))
    return [normalize_candidate(candidate, text) for candidate in candidates]


def query_terms(query):
    """返回最多12个非空关键词，供参数化SQL子串检索；不拼接SQL、不生成向量。"""
    stop = {"什么", "怎么", "如何", "一下", "这个", "那个", "之前", "上次", "请问", "我的", "我们"}
    return list(dict.fromkeys(
        term.strip() for term in jieba.lcut(query[:1000])
        if len(term.strip()) >= 2 and term.strip() not in stop
    ))[:12]


def recall_memories(repository, user_id, query, recent, semantic_index=None, read_plan=None):
    """按问题选择结构化字段并召回本人相关笔记；数量有界，不把全部长期记忆塞给模型。"""
    previous = " ".join(message["content"][:100] for message in recent[-4:] if message["role"] == "user")
    search = query + " " + previous
    keys = ["reply_style"]
    if re.search(r"机器人|设备|型号|充电|保养|维护|故障|它", search):
        keys.append("device_model")
    if re.search(r"拖地|地面|地板|选购|推荐|买", search):
        keys.extend(["floor_type", "pet_status"])
    terms = query_terms(search)
    if read_plan is not None:
        keys = read_plan.fact_keys
        terms = query_terms(query) if read_plan.search_notes else []
        search = read_plan.query or query
    results = repository.search_memories(user_id, terms, keys)
    if semantic_index:
        seen = {row["id"] for row in results}
        results.extend(row for row in semantic_index.search(user_id, search) if row["id"] not in seen)
    entity = next((name for name in ("客厅", "卧室", "书房", "厨房", "阳台") if name in query), None)
    if read_plan is not None and read_plan.entity != "profile":
        entity = read_plan.entity
    if entity:
        results = [row for row in results if row.get("entity_id", "profile") in (entity, "profile")]
    unresolved = repository.unresolved_conflicts(user_id, keys, entity) if hasattr(repository, "unresolved_conflicts") else []
    for conflict in unresolved:
        results.append({"id": "conflict:" + str(conflict["id"]), "memory_key": conflict["memory_key"],
                        "value": "存在互相矛盾的设备陈述，请先询问具体设备/型号，不要猜测。", "source_message_id": conflict["source_message_id"],
                        "needs_clarification": True, "entity_id": conflict["entity_id"]})
    devices = [row for row in results if row.get("attribute", row["memory_key"]) == "device_model"]
    if not entity and len({row.get("entity_id", "profile") for row in devices}) > 1:
        results.append({"id": "clarify:entity", "memory_key": "device_selection", "source_message_id": devices[0]["source_message_id"],
                        "value": "用户有多台设备；若问题未指明对象，先询问是哪台设备。", "needs_clarification": True})
    return sorted(results, key=lambda row: (
        row.get("needs_clarification", False),
        row["memory_key"] in keys,
        sum(term.lower() in row["value"].lower() for term in terms),
        row["source_message_id"],
    ), reverse=True)
