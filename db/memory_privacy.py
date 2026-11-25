"""在调用者已有短事务内提交遗忘；事实、未来计划、摘要与版本同步变化。"""
import json
import re
from uuid import uuid4

from sqlalchemy import func, insert, select, update

from db.memory_extensions import conflicts, forget_epochs, forget_rules, request_epochs, scheduled_facts
from db.memory_schema import conversations, memories, memory_events, messages
from memory.access_policy import MemoryAccessPolicy


def epoch(connection, user_id):
    """读取本用户遗忘版本，旧库尚未遗忘时为0。"""
    return connection.execute(select(forget_epochs.c.version).where(
        forget_epochs.c.user_id == user_id)).scalar_one_or_none() or 0


def policy(connection, user_id):
    """同一连接读取规则及版本；调用方的输出校验处理之后发生的并发变化。"""
    rules = tuple(dict(row) for row in connection.execute(select(forget_rules).where(
        forget_rules.c.user_id == user_id)).mappings())
    return MemoryAccessPolicy(epoch(connection, user_id), rules)


def forget(connection, repository, user_id, candidate, cutoff, now):
    """SQL与自然语言删除共用入口；无现存事实也留下禁止旧历史复活的水位。"""
    from memory.checkpointer import upsert
    entity = "*" if candidate.all_entities else candidate.entity
    all_rows = connection.execute(repository._memory_query().where(memories.c.user_id == user_id)).mappings().all()
    affected = [row for row in all_rows if row["attribute"] == candidate.key
                and entity in ("*", row["entity_id"]) and row["source_message_id"] <= cutoff]
    values = {row["value"] for row in affected}
    ids = [row["id"] for row in affected]
    if ids:
        values.update(connection.execute(select(memory_events.c.value).where(memory_events.c.memory_id.in_(ids))).scalars())
        connection.execute(update(memories).where(memories.c.id.in_(ids)).values(status="deleted", updated_at=now))
        for row in affected:
            connection.execute(insert(memory_events).values(memory_id=row["id"], event="delete", value=row["value"],
                               source_message_id=cutoff, created_at=now))
    if candidate.key.startswith("note:budget:"):
        # 数字别名同样有来源：防止别的旧轮次只复述金额，绕过整条笔记字面匹配。
        values.update(number for value in tuple(values) for number in re.findall(r"\d+(?:\.\d+)?", value))
    pending = connection.execute(select(scheduled_facts).where(
        scheduled_facts.c.user_id == user_id, scheduled_facts.c.status == "pending")).mappings().all()
    for row in pending:
        item = json.loads(row["candidate_json"])
        if item["key"] == candidate.key and entity in ("*", item["entity"]):
            values.add(item["value"])
            connection.execute(update(scheduled_facts).where(scheduled_facts.c.id == row["id"]).values(status="cancelled"))
    query = update(conflicts).where(conflicts.c.user_id == user_id, conflicts.c.memory_key == candidate.key,
                                    conflicts.c.source_message_id <= cutoff)
    if entity != "*":
        query = query.where(conflicts.c.entity_id == entity)
    connection.execute(query.values(status="resolved"))
    connection.execute(insert(forget_rules).values(id=uuid4().hex, user_id=user_id, attribute=candidate.key,
        entity_id=entity, cutoff_id=cutoff, values_json=json.dumps(sorted(values), ensure_ascii=False), created_at=now))
    version = epoch(connection, user_id) + 1
    upsert(connection, forget_epochs, dict(user_id=user_id, version=version))
    # 摘要缺少逐事实来源，保守作废本人所有会话摘要；原文仍保留，之后只从安全原文重建。
    connection.execute(update(conversations).where(conversations.c.user_id == user_id).values(
        summary="", summary_upto=0, revision=conversations.c.revision + 1))
    return True


def record_request(connection, user_id, turn_id):
    """用户操作提交后记录请求版本，恢复或输出时可拒绝过期检查点。"""
    connection.execute(insert(request_epochs).values(turn_id=turn_id, version=epoch(connection, user_id)))


def assert_current(connection, user_id, turn_id):
    """缺乏版本记录的旧请求在发生遗忘后也不得直接恢复。"""
    previous = connection.execute(select(request_epochs.c.version).where(request_epochs.c.turn_id == turn_id)).scalar_one_or_none()
    if (previous or 0) != epoch(connection, user_id):
        raise ValueError("记忆已被删除或更新遗忘范围，本次旧上下文已失效，请重新发送请求")


def migrate_legacy_deletions(repository):
    """幂等补齐旧版软删除的屏障；只增记录/作废旧摘要，不删除原始档案。"""
    from memory.long_term import MemoryCandidate
    with repository.engine.begin() as connection:
        deleted = connection.execute(repository._memory_query().where(memories.c.status == "deleted")).mappings().all()
        changed = set()
        for row in deleted:
            covered = connection.execute(select(forget_rules.c.id).where(forget_rules.c.user_id == row["user_id"],
                forget_rules.c.attribute == row["attribute"], forget_rules.c.cutoff_id >= row["source_message_id"])).first()
            if covered:
                continue
            cutoff = connection.execute(select(func.max(messages.c.id)).join(
                conversations, conversations.c.scope == messages.c.scope).where(
                conversations.c.user_id == row["user_id"], messages.c.created_at <= row["updated_at"])).scalar_one() or row["source_message_id"]
            forget(connection, repository, row["user_id"], MemoryCandidate(row["attribute"], "", "delete", row["entity_id"]),
                   cutoff, row["updated_at"])
            changed.add(row["user_id"])
        for user_id in changed:
            repository._advance_memory_version(connection, user_id)
