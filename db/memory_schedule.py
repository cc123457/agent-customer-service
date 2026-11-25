"""未来事实先存计划，到期才在用户互斥短事务内激活，事实与Redis版本一起提交。"""
import json
from dataclasses import asdict
from uuid import uuid4

from sqlalchemy import func, insert, select, update
from db.memory_extensions import scheduled_facts
from db.memory_schema import memory_users
from memory.long_term import MemoryCandidate
from services.chat_types import SessionCapacityError


def schedule(connection, repository, user_id, message_id, candidate, source, effective_at):
    """在原消息事务中保存有限计划；先保留旧事实，禁止未来值提前替换当前值。"""
    count = connection.execute(select(func.count()).select_from(scheduled_facts).where(
        scheduled_facts.c.user_id == user_id, scheduled_facts.c.status == "pending")).scalar_one()
    if count >= repository.limits["max_active_notes_per_user"]:
        raise SessionCapacityError("未来记忆计划已达配额，请先整理")
    connection.execute(insert(scheduled_facts).values(id=uuid4().hex, user_id=user_id,
        storage_key=candidate.storage_key, source_message_id=message_id, effective_at=effective_at,
        candidate_json=json.dumps(asdict(candidate), ensure_ascii=False), source=source, status="pending"))
    return True


def activate_due(repository, user_id):
    """获取用户写锁后复查计划；删除/更新过的旧计划不复活，成功变更时推进缓存版本。"""
    changed = False
    now = repository.clock()
    with repository.engine.begin() as connection:
        # SQLite的FOR UPDATE不生效，先做等值UPDATE获得写事务锁；MySQL也锁同一个用户行。
        connection.execute(update(memory_users).where(memory_users.c.user_id == user_id).values(user_id=user_id))
        due = connection.execute(select(scheduled_facts).where(scheduled_facts.c.user_id == user_id,
              scheduled_facts.c.status == "pending", scheduled_facts.c.effective_at <= now)
              .order_by(scheduled_facts.c.effective_at, scheduled_facts.c.source_message_id)).mappings().all()
        for row in due:
            applied = repository._apply_memory(connection, user_id, row["source_message_id"],
                       MemoryCandidate(**json.loads(row["candidate_json"])), row["source"], now)
            connection.execute(update(scheduled_facts).where(scheduled_facts.c.id == row["id"]).values(
                status="applied" if applied else "superseded"))
            changed = changed or applied
        if changed:
            new_version = repository._advance_memory_version(connection, user_id)
    if changed and repository.fact_cache:
        repository.fact_cache.note_invalidation(user_id, partition=repository._cache_partition, version=new_version)
