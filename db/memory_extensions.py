"""本轮新增表独立注册，避免直接ALTER已有档案表；事实对象、冲突和冷归档可增量部署。"""
from sqlalchemy import Column, Float, ForeignKey, Index, Integer, String, Table, Text
from db.memory_schema import metadata, memories, messages

fact_details = Table(
    "memory_fact_details", metadata,
    Column("memory_id", String(32), ForeignKey(memories.c.id), primary_key=True),
    Column("attribute", String(64), nullable=False),
    Column("entity_id", String(100), nullable=False),
    Column("evidence", Text, nullable=False),
    Column("valid_at", String(40), nullable=False, default=""),
)
conflicts = Table(
    "memory_conflicts", metadata,
    Column("id", Integer, primary_key=True, autoincrement=True),
    Column("user_id", String(128), nullable=False, index=True),
    Column("memory_key", String(128), nullable=False),
    Column("entity_id", String(100), nullable=False),
    Column("source_message_id", Integer, ForeignKey(messages.c.id), nullable=False),
    Column("evidence", Text, nullable=False),
    Column("status", String(16), nullable=False),
    Column("created_at", Float(precision=53), nullable=False),
)
scheduled_facts = Table(
    "memory_scheduled_facts", metadata,
    Column("id", String(32), primary_key=True),
    Column("user_id", String(128), nullable=False),
    Column("storage_key", String(128), nullable=False),
    Column("source_message_id", Integer, ForeignKey(messages.c.id), nullable=False),
    Column("effective_at", Float(precision=53), nullable=False),
    Column("candidate_json", Text, nullable=False),
    Column("source", Text, nullable=False),
    Column("status", String(16), nullable=False),
    Index("ix_memory_scheduled_due", "user_id", "status", "effective_at"),
)
archive_batches = Table(
    "memory_archive_batches", metadata,
    Column("id", String(32), primary_key=True),
    Column("file_name", String(128), nullable=False),
    Column("sha256", String(64), nullable=False),
    Column("message_count", Integer, nullable=False),
    Column("created_at", Float(precision=53), nullable=False),
)
archived_messages = Table(
    "memory_archived_messages", metadata,
    Column("message_id", Integer, ForeignKey(messages.c.id), primary_key=True),
    Column("batch_id", String(32), ForeignKey(archive_batches.c.id), nullable=False),
)
retired_archives = Table(
    "memory_retired_archives", metadata,
    Column("batch_id", String(32), ForeignKey(archive_batches.c.id), primary_key=True),
    Column("retired_at", Float(precision=53), nullable=False),
)
forget_rules = Table(
    "memory_forget_rules", metadata,
    Column("id", String(32), primary_key=True),
    Column("user_id", String(128), nullable=False, index=True),
    Column("attribute", String(128), nullable=False),
    Column("entity_id", String(100), nullable=False),
    Column("cutoff_id", Integer, nullable=False),
    Column("values_json", Text, nullable=False),
    Column("created_at", Float(precision=53), nullable=False),
)
forget_epochs = Table(
    "memory_forget_epochs", metadata,
    Column("user_id", String(128), primary_key=True),
    Column("version", Integer, nullable=False),
)
request_epochs = Table(
    "memory_request_epochs", metadata,
    Column("turn_id", String(32), primary_key=True),
    Column("version", Integer, nullable=False),
)
for table in (fact_details, conflicts, scheduled_facts, archive_batches, archived_messages, retired_archives,
              forget_rules, forget_epochs, request_epochs):
    table.dialect_options["mysql"].update(engine="InnoDB", charset="utf8mb4", collate="utf8mb4_bin")
