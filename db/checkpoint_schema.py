"""独立的图检查点表；与原文档案共用数据库，保存完整快照和节点待提交写入。"""
from sqlalchemy import Column, Float, Integer, LargeBinary, String, Table
from sqlalchemy.dialects.mysql import LONGBLOB
from db.memory_schema import metadata

blob = LargeBinary().with_variant(LONGBLOB(), "mysql")
checkpoints = Table(
    "agent_checkpoints", metadata,
    Column("thread_id", String(64), primary_key=True),
    Column("namespace", String(256), primary_key=True),
    Column("checkpoint_id", String(64), primary_key=True),
    Column("parent_id", String(64)),
    Column("payload_type", String(32), nullable=False),
    Column("payload", blob, nullable=False),
    Column("metadata_type", String(32), nullable=False),
    Column("metadata", blob, nullable=False),
    Column("created_at", Float(precision=53), nullable=False),
)
checkpoint_writes = Table(
    "agent_checkpoint_writes", metadata,
    Column("thread_id", String(64), primary_key=True),
    Column("namespace", String(256), primary_key=True),
    Column("checkpoint_id", String(64), primary_key=True),
    Column("task_id", String(64), primary_key=True),
    Column("write_index", Integer, primary_key=True),
    Column("task_path", String(512), nullable=False),
    Column("channel", String(256), nullable=False),
    Column("payload_type", String(32), nullable=False),
    Column("payload", blob, nullable=False),
)
for table in (checkpoints, checkpoint_writes):
    table.dialect_options["mysql"].update(engine="InnoDB", charset="utf8mb4", collate="utf8mb4_bin")
