"""历史与长期记忆的独立关系模型；只增建本模块表，不修改演示用户/设备库。"""
from sqlalchemy import Column, Float, ForeignKey, Index, Integer, MetaData, String, Table, Text, UniqueConstraint

metadata = MetaData()
memory_users = Table(
    "memory_users", metadata,
    Column("user_id", String(128), primary_key=True),
)
# 用户级事实版本跨会话共享，与事实写入在同一事务推进；新增表兼容已有档案。
memory_cache_versions = Table(
    "memory_cache_versions", metadata,
    Column("user_id", String(128), ForeignKey(memory_users.c.user_id), primary_key=True),
    Column("version", Integer, nullable=False, default=0),
)
conversations = Table(
    "memory_conversations", metadata,
    Column("scope", String(64), primary_key=True),
    Column("user_id", String(128), nullable=False, index=True),
    Column("session_id", String(128), nullable=False),
    Column("summary", Text, nullable=False, default=""),
    Column("summary_upto", Integer, nullable=False, default=0),
    Column("revision", Integer, nullable=False, default=0),
    Column("owner", String(32), nullable=False, default=""),
    Column("lease_until", Float(precision=53), nullable=False, default=0),
)
turns = Table(
    "memory_turns", metadata,
    Column("id", String(32), primary_key=True),
    Column("scope", String(64), ForeignKey(conversations.c.scope), nullable=False),
    Column("request_id", String(64), nullable=False),
    Column("status", String(16), nullable=False),
    Column("created_at", Float(precision=53), nullable=False),
    UniqueConstraint("scope", "request_id", name="uq_memory_request"),
)
messages = Table(
    "memory_messages", metadata,
    Column("id", Integer, primary_key=True, autoincrement=True),
    Column("turn_id", String(32), ForeignKey(turns.c.id), nullable=False),
    Column("scope", String(64), ForeignKey(conversations.c.scope), nullable=False),
    Column("role", String(16), nullable=False),
    Column("content", Text, nullable=False),
    Column("created_at", Float(precision=53), nullable=False),
    UniqueConstraint("turn_id", "role", name="uq_memory_turn_role"),
    Index("ix_memory_scope_message", "scope", "id"),
)
memories = Table(
    "user_memories", metadata,
    Column("id", String(32), primary_key=True),
    Column("user_id", String(128), nullable=False),
    Column("memory_key", String(128), nullable=False),
    Column("value", Text, nullable=False),
    Column("source_message_id", Integer, ForeignKey(messages.c.id), nullable=False),
    Column("status", String(16), nullable=False),
    Column("updated_at", Float(precision=53), nullable=False),
    UniqueConstraint("user_id", "memory_key", name="uq_user_memory_key"),
)
memory_events = Table(
    "user_memory_events", metadata,
    Column("id", Integer, primary_key=True, autoincrement=True),
    Column("memory_id", String(32), ForeignKey(memories.c.id), nullable=False),
    Column("event", String(16), nullable=False),
    Column("value", Text, nullable=False),
    Column("source_message_id", Integer, nullable=False),
    Column("created_at", Float(precision=53), nullable=False),
)
summary_versions = Table(
    "memory_summary_versions", metadata,
    Column("id", Integer, primary_key=True, autoincrement=True),
    Column("scope", String(64), ForeignKey(conversations.c.scope), nullable=False),
    Column("summary", Text, nullable=False),
    Column("upto_id", Integer, nullable=False),
    Column("created_at", Float(precision=53), nullable=False),
)

for table in metadata.tables.values():
    table.dialect_options["mysql"]["engine"] = "InnoDB"
    table.dialect_options["mysql"]["charset"] = "utf8mb4"
    table.dialect_options["mysql"]["collate"] = "utf8mb4_bin"
