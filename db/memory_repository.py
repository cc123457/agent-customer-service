"""历史是原文档案，摘要只推进水位；支持 MySQL/InnoDB 和本地 SQLite。"""
import hashlib
import time
from pathlib import Path
from uuid import uuid4

from sqlalchemy import LargeBinary, and_, cast, create_engine, func, insert, or_, select, text, update
from sqlalchemy.engine import make_url
from sqlalchemy.exc import IntegrityError, SQLAlchemyError

from db.memory_schema import conversations, memories, memory_cache_versions, memory_events, memory_users, messages, metadata, summary_versions, turns
from db.memory_extensions import conflicts, fact_details, scheduled_facts
from memory.fact_rules import current_rows, effective_timestamp, normalize_candidate, note_key
from memory.fact_cache import FACT_FIELDS
from services.chat_types import SessionCapacityError, SessionLeaseError


class DuplicateRequestError(ValueError):
    """同一用户会话重复提交已存在的 request_id，调用方应查询既有记录而不是重复追加。"""


class MemoryRepository:
    """关系存储入口；方法各自完成短事务，不在模型等待期间持有数据库事务。"""

    def __init__(self, database_url, clock=time.time, limits=None, fact_cache=None):
        """输入显式数据库 URL；仅接受 SQLite 或 MySQL，初始化连接池但不删除/迁移原有表。"""
        url = make_url(database_url)
        if url.get_backend_name() not in ("sqlite", "mysql"):
            raise ValueError("记忆数据库仅支持 SQLite 或 MySQL")
        options = {"pool_pre_ping": True}
        if url.get_backend_name() == "sqlite":
            if url.database and url.database != ":memory:":
                Path(url.database).parent.mkdir(parents=True, exist_ok=True)
            options["connect_args"] = {"check_same_thread": False, "timeout": 10}
        else:
            options["connect_args"] = {"connect_timeout": 3, "read_timeout": 5, "write_timeout": 5}
        self.engine = create_engine(url, **options)
        self.fact_cache = fact_cache
        identity = url.render_as_string(hide_password=False)
        if url.get_backend_name() == "sqlite" and url.database in (None, "", ":memory:"):
            identity += uuid4().hex
        self._cache_partition = hashlib.sha256(identity.encode("utf-8")).hexdigest()
        self.clock = clock
        self.limits = dict(max_hot_bytes_per_user=10000000, max_active_notes_per_user=200, max_answer_bytes=32000)
        self.limits.update(limits or {})
        if any(value <= 0 for value in self.limits.values()):
            raise ValueError("记忆配额必须大于0")

    def initialize(self):
        """增建缺失记忆表；已有MySQL身份列必须区分大小写，否则拒绝启动，要求显式迁移。"""
        from db import checkpoint_schema, memory_extensions  # 增量注册新表，不修改已有档案表。
        metadata.create_all(self.engine)
        from db.memory_privacy import migrate_legacy_deletions
        migrate_legacy_deletions(self)
        if self.engine.dialect.name == "mysql":
            with self.engine.connect() as connection:
                rows = connection.execute(text(
                    "SELECT TABLE_NAME, COLLATION_NAME FROM information_schema.COLUMNS "
                    "WHERE TABLE_SCHEMA=DATABASE() AND COLUMN_NAME='user_id' "
                    "AND TABLE_NAME IN ('memory_users','memory_conversations','user_memories')"
                )).all()
            if len(rows) != 3 or any(row[1] != "utf8mb4_bin" for row in rows):
                raise SQLAlchemyError("记忆库user_id排序规则不安全，请先显式迁移为utf8mb4_bin；未修改已有数据")

    @staticmethod
    def scope_key(scope):
        """将服务端生成的用户/会话 scope 哈希为定长数据库键，不用哈希代替授权。"""
        return hashlib.sha256(scope.encode("utf-8")).hexdigest()

    def _ensure(self, scope, user_id, session_id):
        """首次访问时补齐用户互斥行与会话行；唯一约束处理并发创建，不重置已有摘要。"""
        key = self.scope_key(scope)
        for table, identity, values in (
            (memory_users, memory_users.c.user_id == user_id, {"user_id": user_id}),
            (conversations, conversations.c.scope == key, {
                "scope": key, "user_id": user_id, "session_id": session_id,
                "summary": "", "summary_upto": 0, "revision": 0, "owner": "", "lease_until": 0,
            }),
        ):
            try:
                with self.engine.begin() as connection:
                    if connection.execute(select(table).where(identity)).first() is None:
                        connection.execute(insert(table).values(**values))
            except IntegrityError:
                with self.engine.connect() as connection:
                    if connection.execute(select(table).where(identity)).first() is None:
                        raise
        return key

    def claim(self, scope, user_id, session_id, owner, ttl):
        """申请数据库中的会话租约，配合 Redis 租约防止缓存丢失后发生并发写；返回 bool。"""
        key = self._ensure(scope, user_id, session_id)
        now = self.clock()
        with self.engine.begin() as connection:
            result = connection.execute(update(conversations).where(
                conversations.c.scope == key, conversations.c.user_id == user_id,
                conversations.c.lease_until <= now,
            ).values(owner=owner, lease_until=now + ttl))
            if result.rowcount == 1:
                connection.execute(update(turns).where(
                    turns.c.scope == key, turns.c.status == "running",
                ).values(status="failed"))
            return result.rowcount == 1

    def _owned(self, connection, key, owner):
        """事务内锁定并返回会话行；只有未过期的匹配 owner 可写，失效则抛异常。"""
        row = connection.execute(select(conversations).where(
            conversations.c.scope == key,
        ).with_for_update()).mappings().one()
        if row["owner"] != owner or row["lease_until"] <= self.clock():
            raise SessionLeaseError("数据库会话处理权已失效")
        return row

    def release(self, scope, owner):
        """释放匹配owner的租约，并将其遗留running请求标失败；不删除历史，正常完成状态不变。"""
        key = self.scope_key(scope)
        with self.engine.begin() as connection:
            result = connection.execute(update(conversations).where(
                conversations.c.scope == key, conversations.c.owner == owner,
            ).values(owner="", lease_until=0))
            if result.rowcount == 1:
                connection.execute(update(turns).where(
                    turns.c.scope == key, turns.c.status == "running",
                ).values(status="failed"))
            return result.rowcount == 1

    def start_request(self, scope, user_id, owner, request_id, text, candidates, expected_forget_version=None):
        """原子保存用户原文、running 请求及规则提取的记忆；拒绝重复请求ID，返回内部请求与消息ID。"""
        key = self.scope_key(scope)
        turn_id = uuid4().hex
        with self.engine.begin() as connection:
            # SQLite也取得写锁；不同会话共享本人事实/遗忘版本，不能仅锁会话行。
            connection.execute(update(memory_users).where(memory_users.c.user_id == user_id).values(user_id=user_id))
            from db.memory_privacy import epoch
            if expected_forget_version is not None and epoch(connection, user_id) != expected_forget_version:
                raise ValueError("分析期间遗忘规则已变化，请重新发送请求")
            row = self._owned(connection, key, owner)
            if row["user_id"] != user_id:
                raise PermissionError("会话不属于当前用户")
            connection.execute(select(memory_users).where(
                memory_users.c.user_id == user_id,
            ).with_for_update()).one()
            existing = connection.execute(select(turns.c.id).where(
                turns.c.scope == key, turns.c.request_id == request_id,
            )).first()
            if existing:
                raise DuplicateRequestError("request_id 已存在，请查询历史确认结果，不要重复追加")
            self._check_hot_quota(connection, user_id, text)
            now = self.clock()
            connection.execute(insert(turns).values(
                id=turn_id, scope=key, request_id=request_id, status="running", created_at=now,
            ))
            message_id = connection.execute(insert(messages).values(
                turn_id=turn_id, scope=key, role="user", content=text, created_at=now,
            )).inserted_primary_key[0]
            changed = False
            for candidate in candidates:
                changed = self._apply_memory(connection, user_id, message_id, candidate, text, now) or changed
            from db.memory_privacy import record_request
            record_request(connection, user_id, turn_id)
            if changed:
                new_version = self._advance_memory_version(connection, user_id)
            connection.execute(update(conversations).where(conversations.c.scope == key).values(
                revision=conversations.c.revision + 1,
            ))
        if changed and self.fact_cache:
            self.fact_cache.note_invalidation(user_id, partition=self._cache_partition, version=new_version)
        return turn_id, message_id

    def _apply_memory(self, connection, user_id, message_id, candidate, source, now):
        """保存有原文依据的候选，保留更正事件和删除墓碑。

        返回是否改变事实/冲突状态，供start_request在同一事务里推进缓存版本。
        """
        candidate = normalize_candidate(candidate, source)
        if candidate.action not in ("upsert", "delete", "conflict"):
            raise ValueError("记忆操作无效")
        if candidate.action != "delete" and candidate.value not in source:
            raise ValueError("长期记忆值必须直接来自当前用户原文")
        if candidate.action != "delete" and candidate.entity != "profile" and candidate.entity not in source:
            raise ValueError("记忆对象必须有当前原文依据")
        if candidate.evidence and candidate.evidence not in source:
            raise ValueError("记忆证据必须是用户原文片段")
        if candidate.valid_at and candidate.valid_at not in source:
            raise ValueError("有效时间必须来自原文，不能推测")
        if candidate.action == "delete":
            from db.memory_privacy import forget
            return forget(connection, self, user_id, candidate, message_id, now)
        from db.memory_extensions import forget_rules
        if connection.execute(select(forget_rules.c.id).where(
            forget_rules.c.user_id == user_id, forget_rules.c.attribute == candidate.key,
            forget_rules.c.entity_id.in_(["*", candidate.entity]), forget_rules.c.cutoff_id >= message_id,
        ).limit(1)).first():
            return False
        key = candidate.storage_key
        starts = effective_timestamp(candidate.valid_at)
        if starts is not None and starts > now and candidate.action != "delete":
            from db.memory_schedule import schedule
            return schedule(connection, self, user_id, message_id, candidate, source, starts)
        row = connection.execute(select(memories).where(
            memories.c.user_id == user_id, memories.c.memory_key == key,
        )).mappings().first()
        if row and row["source_message_id"] >= message_id:
            return False
        if candidate.key.startswith("note:budget:"):
            legacy = connection.execute(self._memory_query().where(memories.c.user_id == user_id,
                     memories.c.status == "active", memories.c.memory_key != key)).mappings().all()
            for previous in legacy:
                if (previous.get("entity_id") == candidate.entity and previous.get("attribute", "").startswith("note:")
                        and note_key(previous["value"]) == candidate.key and previous["source_message_id"] < message_id):
                    connection.execute(update(memories).where(memories.c.id == previous["id"]).values(status="superseded"))
        if candidate.key.startswith("note:") and (row is None or row["status"] != "active"):
            count = connection.execute(select(func.count()).select_from(memories).where(
                memories.c.user_id == user_id, memories.c.status == "active", memories.c.memory_key.startswith("note:")
            )).scalar_one()
            if count >= self.limits["max_active_notes_per_user"]:
                raise SessionCapacityError("有效记忆笔记数量已达配额，请先整理已有笔记")
        if candidate.action == "conflict":
            connection.execute(insert(conflicts).values(
                user_id=user_id, memory_key=candidate.key, entity_id=candidate.entity,
                source_message_id=message_id, evidence=source, status="unresolved", created_at=now,
            ))
            if row:
                connection.execute(update(memories).where(memories.c.id == row["id"]).values(status="conflicted"))
            return True
        connection.execute(update(conflicts).where(
            conflicts.c.user_id == user_id, conflicts.c.memory_key == candidate.key,
            conflicts.c.entity_id == candidate.entity, conflicts.c.status == "unresolved",
        ).values(status="resolved"))
        memory_id = row["id"] if row else uuid4().hex
        value = candidate.value
        values = {
            "value": value, "source_message_id": message_id,
            "status": "active", "updated_at": now,
        }
        if row:
            connection.execute(update(memories).where(memories.c.id == memory_id).values(**values))
        else:
            connection.execute(insert(memories).values(
                id=memory_id, user_id=user_id, memory_key=key, **values,
            ))
        from memory.checkpointer import upsert
        upsert(connection, fact_details, dict(memory_id=memory_id, attribute=candidate.key,
                                            entity_id=candidate.entity, evidence=candidate.evidence or candidate.value,
                                            valid_at=candidate.valid_at))
        connection.execute(insert(memory_events).values(
            memory_id=memory_id, event=candidate.action, value=value,
            source_message_id=message_id, created_at=now,
        ))
        return True

    def _advance_memory_version(self, connection, user_id):
        """同事务推进并返回新版本，提交后据此回收旧键；回滚时事实与版本一起回滚。"""
        result = connection.execute(update(memory_cache_versions).where(
            memory_cache_versions.c.user_id == user_id,
        ).values(version=memory_cache_versions.c.version + 1))
        if result.rowcount == 0:
            connection.execute(insert(memory_cache_versions).values(user_id=user_id, version=1))
        return connection.execute(select(memory_cache_versions.c.version).where(
            memory_cache_versions.c.user_id == user_id)).scalar_one()

    def memory_version(self, user_id):
        """一次SELECT同时查缓存版本与最近生效点；跨过日期时先原子激活，再读取新版本。"""
        due = select(func.min(scheduled_facts.c.effective_at)).where(
            scheduled_facts.c.user_id == user_id, scheduled_facts.c.status == "pending").scalar_subquery()
        with self.engine.connect() as connection:
            row = connection.execute(select(memory_cache_versions.c.version, due.label("next_due")).where(
                memory_cache_versions.c.user_id == user_id,
            )).first()
        if row and row.next_due is not None and row.next_due <= self.clock():
            from db.memory_schedule import activate_due
            activate_due(self, user_id)
            return self.memory_version(user_id)
        return row.version if row else 0

    def memory_cache_stats(self, user_id):
        """API读取本人当前进程缓存统计；禁用时明确返回不适用。"""
        if self.fact_cache is None:
            return {"enabled": False, "available": False, "scope": "current_process", "hit_rate": None}
        return self.fact_cache.stats(user_id)

    def snapshot(self, scope, owner, keep=10, pending_limit=200):
        """事务内读取摘要水位、有限早期原文与近期消息；原文存档不因摘要成功而删除。"""
        key = self.scope_key(scope)
        with self.engine.begin() as connection:
            row = dict(self._owned(connection, key, owner))
            eligible = and_(
                messages.c.scope == key,
                or_(turns.c.status == "completed", turns.c.status == "running"),
            )
            base = select(messages).join(turns, turns.c.id == messages.c.turn_id).where(eligible)
            pending = connection.execute(base.where(
                messages.c.id > row["summary_upto"],
            ).order_by(messages.c.id).limit(pending_limit)).mappings().all()
            recent = connection.execute(base.order_by(
                messages.c.id.desc(),
            ).limit(keep)).mappings().all()
            row["pending"] = [dict(message) for message in pending]
            row["recent"] = list(reversed([dict(message) for message in recent]))
            from db.memory_privacy import epoch
            row["forget_version"] = epoch(connection, row["user_id"])
            return row

    def access_policy(self, user_id):
        """读取本人持久遗忘规则，供近期窗口、摘要和两类检索共同执行。"""
        from db.memory_privacy import policy
        with self.engine.connect() as connection:
            return policy(connection, user_id)

    def deletion_targets(self, user_id):
        """删除定位同时考虑尚未生效计划；相同属性/对象合并，不向模型暴露任意数据库键。"""
        import json
        rows = self.active_memories(user_id)
        with self.engine.connect() as connection:
            pending = connection.execute(select(scheduled_facts).where(
                scheduled_facts.c.user_id == user_id, scheduled_facts.c.status == "pending")).mappings()
            for row in pending:
                item = json.loads(row["candidate_json"])
                rows.append(dict(memory_key=item["key"], attribute=item["key"], value=item["value"],
                                 entity_id=item["entity"], source_message_id=row["source_message_id"]))
        return list({(row["attribute"], row["entity_id"]): row for row in rows}.values())

    def assert_request_current(self, user_id, turn_id):
        """输出/恢复前复核SQL版本；不能让旧检查点重新恢复已遗忘的上下文。"""
        from db.memory_privacy import assert_current
        with self.engine.connect() as connection:
            assert_current(connection, user_id, turn_id)

    def commit_summary(self, scope, owner, revision, upto_id, summary):
        """版本和租约匹配后保存摘要及覆盖水位，并留摘要版本；不会 DELETE 原文。"""
        key = self.scope_key(scope)
        with self.engine.begin() as connection:
            row = self._owned(connection, key, owner)
            if row["revision"] != revision or upto_id <= row["summary_upto"]:
                return False
            covered = connection.execute(select(messages.c.id).where(
                messages.c.scope == key, messages.c.id == upto_id,
            )).first()
            if covered is None or not summary.strip():
                raise ValueError("摘要水位或正文无效")
            connection.execute(update(conversations).where(conversations.c.scope == key).values(
                summary=summary, summary_upto=upto_id, revision=revision + 1,
            ))
            connection.execute(insert(summary_versions).values(
                scope=key, summary=summary, upto_id=upto_id, created_at=self.clock(),
            ))
            return True

    def finish_request(self, scope, owner, turn_id, answer):
        """正常结束时事务保存助手回答并标记completed；失效owner不能写入，异常会回滚。"""
        key = self.scope_key(scope)
        with self.engine.begin() as connection:
            conversation = self._owned(connection, key, owner)
            connection.execute(update(memory_users).where(memory_users.c.user_id == conversation["user_id"]).values(user_id=conversation["user_id"]))
            from db.memory_privacy import assert_current
            assert_current(connection, conversation["user_id"], turn_id)
            connection.execute(select(memory_users).where(memory_users.c.user_id == conversation["user_id"]).with_for_update()).one()
            if len(answer.encode("utf-8")) > self.limits["max_answer_bytes"]:
                raise SessionCapacityError("回答超过归档长度上限")
            self._check_hot_quota(connection, conversation["user_id"], answer)
            row = connection.execute(select(turns).where(
                turns.c.id == turn_id, turns.c.scope == key,
            )).mappings().one()
            if row["status"] != "running":
                raise DuplicateRequestError("该请求已经结束")
            connection.execute(insert(messages).values(
                scope=key, turn_id=turn_id, role="assistant", content=answer, created_at=self.clock(),
            ))
            connection.execute(update(turns).where(turns.c.id == turn_id).values(status="completed"))
            connection.execute(update(conversations).where(conversations.c.scope == key).values(
                revision=conversations.c.revision + 1,
            ))

    def _check_hot_quota(self, connection, user_id, added_text):
        """在用户互斥行锁内统计热正文UTF-8字节；超额拒绝新增，不静默删除旧消息。"""
        used = connection.execute(select(func.coalesce(func.sum(func.length(cast(messages.c.content, LargeBinary))), 0)).select_from(
            messages.join(conversations, conversations.c.scope == messages.c.scope)
        ).where(conversations.c.user_id == user_id)).scalar_one()
        if used + len(added_text.encode("utf-8")) > self.limits["max_hot_bytes_per_user"]:
            raise SessionCapacityError("聊天热正文已达存储配额，请先归档或整理历史")

    def fail_request(self, turn_id):
        """异常/取消后将仍在running的本请求标成failed，不伪造助手回答，也不删除用户原文。"""
        with self.engine.begin() as connection:
            connection.execute(update(turns).where(
                turns.c.id == turn_id, turns.c.status == "running",
            ).values(status="failed"))

    def request(self, scope, user_id, request_id):
        """按已认证用户与会话读取请求；用于恢复和检查点列表，其他用户与不存在均返回None。"""
        with self.engine.connect() as connection:
            row = connection.execute(select(turns).join(conversations, conversations.c.scope == turns.c.scope).where(
                turns.c.scope == self.scope_key(scope), conversations.c.user_id == user_id,
                turns.c.request_id == request_id,
            )).mappings().first()
            return dict(row) if row else None

    def resume_request(self, scope, user_id, owner, request_id):
        """在持有新租约时重开本会话最后一个失败请求，不重复归档用户消息或提取长期记忆。"""
        key = self.scope_key(scope)
        with self.engine.begin() as connection:
            self._owned(connection, key, owner)
            row = connection.execute(select(turns).join(conversations, conversations.c.scope == turns.c.scope).where(
                turns.c.scope == key, turns.c.request_id == request_id, conversations.c.user_id == user_id,
            )).mappings().first()
            if row is None:
                raise LookupError("未找到本人请求")
            from db.memory_privacy import assert_current
            assert_current(connection, user_id, row["id"])
            source = connection.execute(select(messages.c.id).where(
                messages.c.turn_id == row["id"], messages.c.role == "user",
            )).scalar_one()
            latest = connection.execute(select(messages.c.turn_id).where(
                messages.c.scope == key, messages.c.role == "user",
            ).order_by(messages.c.id.desc()).limit(1)).scalar_one()
            if row["status"] != "failed" or latest != row["id"]:
                raise DuplicateRequestError("只能恢复会话最后一个失败请求；已有后续对话请新建请求")
            connection.execute(update(turns).where(turns.c.id == row["id"]).values(status="running"))
            return row["id"], source

    def history(self, scope, user_id, limit=50, before_id=None):
        """按认证用户和会话分页返回原始记录及请求状态，不能借session_id读取其他用户。"""
        statement = select(messages, turns.c.status, turns.c.request_id).join(
            turns, turns.c.id == messages.c.turn_id,
        ).join(conversations, conversations.c.scope == messages.c.scope).where(
            messages.c.scope == self.scope_key(scope), conversations.c.user_id == user_id,
        )
        if before_id is not None:
            statement = statement.where(messages.c.id < before_id)
        with self.engine.connect() as connection:
            return [dict(row) for row in connection.execute(
                statement.order_by(messages.c.id.desc()).limit(min(max(limit, 1), 100)),
            ).mappings()]

    def search_memories(self, user_id, terms=(), keys=(), limit=20):
        """本人结构化事实可走Redis；笔记关键词查询仍走SQL，筛选、排序与数量限制不变。"""
        rows = []
        version = self.memory_version(user_id)
        if keys:
            def load_facts():
                """缓存未命中时执行原有SQL字段查询；连接仅在实际读取期间占用。"""
                with self.engine.connect() as connection:
                    query = self._memory_query().where(
                        memories.c.user_id == user_id, memories.c.status == "active",
                        func.coalesce(fact_details.c.attribute, memories.c.memory_key).in_(keys),
                    )
                    return [dict(row) for row in connection.execute(
                        query.order_by(memories.c.source_message_id.desc()).limit(min(limit, 100))
                    ).mappings()]

            if self.fact_cache is not None and set(keys).issubset(FACT_FIELDS) and limit > 0:
                rows = self.fact_cache.get_or_load(
                    self._cache_partition, user_id, version, keys,
                    min(limit, 100), load_facts,
                )
            else:
                rows = load_facts()
        remaining = min(limit, 100) - len(rows)
        if terms and remaining > 0:
            with self.engine.connect() as connection:
                base = self._memory_query().where(memories.c.user_id == user_id, memories.c.status == "active")
                attribute = func.coalesce(fact_details.c.attribute, memories.c.memory_key)
                matched = connection.execute(base.where(
                    attribute.not_in(keys),
                    or_(*(func.lower(memories.c.value).contains(term.lower(), autoescape=True) for term in terms)),
                ).order_by(memories.c.source_message_id.desc()).limit(1000)).mappings()
                rows.extend(current_rows([dict(row) for row in matched])[:remaining])
        return self.access_policy(user_id).facts(current_rows(rows))

    def list_memories(self, user_id, limit=50):
        """返回当前用户的有效记忆，供管理接口查看；不返回其他用户或已删除记录。"""
        self.memory_version(user_id)
        with self.engine.connect() as connection:
            return current_rows([dict(row) for row in connection.execute(self._memory_query().where(
                memories.c.user_id == user_id, memories.c.status == "active",
            ).order_by(memories.c.updated_at.desc()).limit(1000)).mappings()])[:min(limit, 100)]

    @staticmethod
    def _memory_query():
        """给旧字段补默认profile对象，新字段返回对象/证据/有效时间，兼容已有记忆行。"""
        return select(memories, func.coalesce(fact_details.c.attribute, memories.c.memory_key).label("attribute"),
                      func.coalesce(fact_details.c.entity_id, "profile").label("entity_id"),
                      fact_details.c.evidence, fact_details.c.valid_at).outerjoin(fact_details, fact_details.c.memory_id == memories.c.id)

    def active_memories(self, user_id, ids=None):
        """向量索引和召回复核读取SQL当前有效事实，强制本人范围，不信任索引中可能过期的内容。"""
        self.memory_version(user_id)
        query = self._memory_query().where(memories.c.user_id == user_id, memories.c.status == "active")
        if ids is not None:
            if not ids:
                return []
        with self.engine.connect() as connection:
            rows = current_rows([dict(row) for row in connection.execute(query.order_by(memories.c.source_message_id.desc()).limit(1000)).mappings()])
        return self.access_policy(user_id).facts([row for row in rows if ids is None or row["id"] in ids])

    def unresolved_conflicts(self, user_id, keys, entity=None):
        """召回前读取本人未解决冲突；有明确对象时只返回该对象，不把冲突当作已确认事实。"""
        query = select(conflicts).where(conflicts.c.user_id == user_id, conflicts.c.status == "unresolved",
                                         conflicts.c.memory_key.in_(keys))
        if entity:
            query = query.where(conflicts.c.entity_id.in_([entity, "profile"]))
        with self.engine.connect() as connection:
            return [dict(row) for row in connection.execute(query.order_by(conflicts.c.id.desc()).limit(10)).mappings()]

    def superseded_values(self, user_id):
        """返回已更正/停用且不再属于任何有效事实的旧原文值，用于遮蔽摘要中的已知旧值。"""
        active = {row["value"] for row in self.active_memories(user_id)}
        with self.engine.connect() as connection:
            old = connection.execute(select(memory_events.c.value).join(memories, memories.c.id == memory_events.c.memory_id).where(
                memories.c.user_id == user_id,
            ).order_by(memory_events.c.id.desc()).limit(1000)).scalars()
            return sorted({value for value in old if value and value not in active}, key=len, reverse=True)

    def delete_memory(self, user_id, memory_id):
        """按认证用户软删除记忆并记录事件；不存在/不属于本人均返回False，不泄露归属。"""
        with self.engine.begin() as connection:
            connection.execute(update(memory_users).where(memory_users.c.user_id == user_id).values(user_id=user_id))
            connection.execute(select(memory_users).where(
                memory_users.c.user_id == user_id,
            ).with_for_update()).first()
            row = connection.execute(select(memories).where(
                memories.c.id == memory_id, memories.c.user_id == user_id,
                memories.c.status == "active",
            ).with_for_update()).mappings().first()
            if row is None:
                return False
            from db.memory_privacy import forget
            from memory.long_term import MemoryCandidate
            details = connection.execute(self._memory_query().where(memories.c.id == memory_id)).mappings().one()
            cutoff = connection.execute(select(func.max(messages.c.id)).join(
                conversations, conversations.c.scope == messages.c.scope).where(conversations.c.user_id == user_id)).scalar_one() or 0
            forget(connection, self, user_id, MemoryCandidate(details["attribute"], "", "delete", details["entity_id"]), cutoff, self.clock())
            new_version = self._advance_memory_version(connection, user_id)
        if self.fact_cache:
            self.fact_cache.note_invalidation(user_id, partition=self._cache_partition, version=new_version)
        return True

    def history_turns(self, user_id, *, session_id=None, after=None, before=None,
                      limit=200, exclude_turn_ids=(), turn_ids=None):
        """先按认证用户/会话/轮次起始时间过滤，再取有限已完成问答；不按原文LIKE匹配。

        before是排他上界；用户消息ID决定新旧次序。返回双方消息供BM25、向量索引和
        命中后SQL复核使用；冷正文仍由MemoryArchive恢复，失败轮次不会成为检索证据。
        """
        if not user_id or not user_id.strip() or limit < 1:
            raise ValueError("历史检索用户和候选上限无效")
        if after is not None and before is not None and after >= before:
            raise ValueError("历史检索开始时间必须早于结束时间")
        if turn_ids is not None and not turn_ids:
            return []
        query = select(turns.c.id, turns.c.request_id, turns.c.created_at,
                       conversations.c.session_id).join(
            conversations, conversations.c.scope == turns.c.scope,
        ).join(messages, messages.c.turn_id == turns.c.id).where(
            conversations.c.user_id == user_id, turns.c.status == "completed", messages.c.role == "user",
        )
        if session_id is not None:
            query = query.where(conversations.c.session_id == session_id)
        if after is not None:
            query = query.where(turns.c.created_at >= after)
        if before is not None:
            query = query.where(turns.c.created_at < before)
        if exclude_turn_ids:
            query = query.where(turns.c.id.not_in(exclude_turn_ids))
        if turn_ids is not None:
            query = query.where(turns.c.id.in_(turn_ids))
        with self.engine.connect() as connection:
            selected = [dict(row) for row in connection.execute(
                query.order_by(messages.c.id.desc()).limit(limit)
            ).mappings()]
            if not selected:
                return []
            ids = [row["id"] for row in selected]
            originals = connection.execute(select(messages).join(
                conversations, conversations.c.scope == messages.c.scope,
            ).where(conversations.c.user_id == user_id, messages.c.turn_id.in_(ids))
                .order_by(messages.c.id)).mappings()
            grouped = {identity: [] for identity in ids}
            for row in originals:
                grouped[row["turn_id"]].append(dict(row))
        return [dict(turn_id=row["id"], request_id=row["request_id"],
                     session_id=row["session_id"], created_at=row["created_at"],
                     messages=grouped[row["id"]]) for row in selected
                if [message["role"] for message in grouped[row["id"]]] == ["user", "assistant"]]
