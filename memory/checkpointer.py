"""LangGraph BaseCheckpointSaver 的关系库实现：完整快照、pending writes、父链和异步接口。"""
import asyncio
import time

from langgraph.checkpoint.base import BaseCheckpointSaver, CheckpointTuple, WRITES_IDX_MAP, get_checkpoint_metadata
from sqlalchemy import delete, select
from sqlalchemy.dialects.mysql import insert as mysql_insert
from sqlalchemy.dialects.sqlite import insert as sqlite_insert

from db.checkpoint_schema import checkpoints, checkpoint_writes


def upsert(connection, table, values, replace=True):
    """按方言执行原子UPSERT；普通节点写入只保留首次结果，特殊错误/中断写入允许更新。"""
    keys = [column.name for column in table.primary_key]
    if connection.dialect.name == "mysql":
        statement = mysql_insert(table).values(**values)
        updates = {key: statement.inserted[key] for key in values if key not in keys} if replace else {keys[0]: table.c[keys[0]]}
        connection.execute(statement.on_duplicate_key_update(**updates))
    else:
        statement = sqlite_insert(table).values(**values)
        if replace:
            statement = statement.on_conflict_do_update(index_elements=keys, set_={
                key: statement.excluded[key] for key in values if key not in keys
            })
        else:
            statement = statement.on_conflict_do_nothing(index_elements=keys)
        connection.execute(statement)


class SqlCheckpointSaver(BaseCheckpointSaver):
    """只存图状态，不授予访问权限；HTTP/服务层先按JWT校验请求归属，不能接受任意thread_id。"""

    def __init__(self, engine):
        """复用关系库连接池与默认安全序列化器，不启用pickle回退；表由MemoryRepository.initialize创建。"""
        super().__init__()
        self.engine = engine

    @staticmethod
    def _keys(config):
        """解析服务端配置，拒绝超长键，不截断以免两个执行线程发生碰撞。"""
        values = config["configurable"]
        thread, namespace = values["thread_id"], values.get("checkpoint_ns", "")
        if not thread or len(thread) > 64 or len(namespace) > 256:
            raise ValueError("检查点线程或命名空间无效")
        return thread, namespace

    @staticmethod
    def _config(thread, namespace, checkpoint_id):
        """构造LangGraph定位配置；checkpoint_id与thread_id共同定位，不能跨线程借用快照。"""
        return {"configurable": {"thread_id": thread, "checkpoint_ns": namespace, "checkpoint_id": checkpoint_id}}

    def _tuple(self, connection, row):
        """读取同检查点的待提交节点输出，按框架要求排序，组装可恢复状态和父快照定位。"""
        identity = (checkpoint_writes.c.thread_id == row["thread_id"],
                    checkpoint_writes.c.namespace == row["namespace"],
                    checkpoint_writes.c.checkpoint_id == row["checkpoint_id"])
        writes = connection.execute(select(checkpoint_writes).where(*identity).order_by(
            checkpoint_writes.c.task_path, checkpoint_writes.c.task_id, checkpoint_writes.c.write_index,
        )).mappings().all()
        return CheckpointTuple(
            self._config(row["thread_id"], row["namespace"], row["checkpoint_id"]),
            self.serde.loads_typed((row["payload_type"], row["payload"])),
            self.serde.loads_typed((row["metadata_type"], row["metadata"])),
            self._config(row["thread_id"], row["namespace"], row["parent_id"]) if row["parent_id"] else None,
            [(item["task_id"], item["channel"], self.serde.loads_typed((item["payload_type"], item["payload"]))) for item in writes],
        )

    def get_tuple(self, config):
        """读取指定或最新完整检查点；不存在返回None，底层故障直接抛出，不伪装成空状态。"""
        thread, namespace = self._keys(config)
        query = select(checkpoints).where(checkpoints.c.thread_id == thread, checkpoints.c.namespace == namespace)
        checkpoint_id = config["configurable"].get("checkpoint_id")
        if checkpoint_id:
            query = query.where(checkpoints.c.checkpoint_id == checkpoint_id)
        with self.engine.connect() as connection:
            row = connection.execute(query.order_by(checkpoints.c.checkpoint_id.desc()).limit(1)).mappings().first()
            return self._tuple(connection, row) if row else None

    def list(self, config, *, filter=None, before=None, limit=None):
        """倒序列出检查点，支持线程、命名空间、父游标和元数据过滤；完整保留父链供回放。"""
        query = select(checkpoints)
        if config:
            query = query.where(checkpoints.c.thread_id == config["configurable"]["thread_id"])
            if "checkpoint_ns" in config["configurable"]:
                query = query.where(checkpoints.c.namespace == config["configurable"]["checkpoint_ns"])
            if config["configurable"].get("checkpoint_id"):
                query = query.where(checkpoints.c.checkpoint_id == config["configurable"]["checkpoint_id"])
        if before:
            query = query.where(checkpoints.c.checkpoint_id < before["configurable"]["checkpoint_id"])
        if limit is not None and limit <= 0:
            return
        count = 0
        with self.engine.connect() as connection:
            for row in connection.execute(query.order_by(checkpoints.c.checkpoint_id.desc())).mappings():
                item = self._tuple(connection, row)
                if filter and any(item.metadata.get(key) != value for key, value in filter.items()):
                    continue
                yield item
                count += 1
                if limit is not None and count >= limit:
                    break

    def put(self, config, checkpoint, metadata, new_versions):
        """短事务保存完整快照和父ID；不依赖外部原文表，不跨模型调用持有事务。"""
        thread, namespace = self._keys(config)
        kind, payload = self.serde.dumps_typed(checkpoint)
        metadata_kind, metadata_payload = self.serde.dumps_typed(get_checkpoint_metadata(config, metadata))
        values = dict(thread_id=thread, namespace=namespace, checkpoint_id=checkpoint["id"],
                      parent_id=config["configurable"].get("checkpoint_id"), payload_type=kind, payload=payload,
                      metadata_type=metadata_kind, metadata=metadata_payload, created_at=time.time())
        with self.engine.begin() as connection:
            upsert(connection, checkpoints, values)
        return self._config(thread, namespace, checkpoint["id"])

    def put_writes(self, config, writes, task_id, task_path=""):
        """保留成功任务的首次输出，允许错误/interrupt等保留槽更新，支持失败超步的继续执行。"""
        thread, namespace = self._keys(config)
        with self.engine.begin() as connection:
            for index, (channel, value) in enumerate(writes):
                kind, payload = self.serde.dumps_typed(value)
                upsert(connection, checkpoint_writes, dict(
                    thread_id=thread, namespace=namespace, checkpoint_id=config["configurable"]["checkpoint_id"],
                    task_id=task_id, write_index=WRITES_IDX_MAP.get(channel, index), task_path=task_path,
                    channel=channel, payload_type=kind, payload=payload,
                ), replace=channel in WRITES_IDX_MAP)

    def delete_thread(self, thread_id):
        """删除整个线程的检查点与写入；不裁剪单个祖先，避免破坏增量通道父链。"""
        with self.engine.begin() as connection:
            connection.execute(delete(checkpoint_writes).where(checkpoint_writes.c.thread_id == thread_id))
            connection.execute(delete(checkpoints).where(checkpoints.c.thread_id == thread_id))

    async def aget_tuple(self, config):
        """异步读取通过工作线程执行SQL，避免同步数据库驱动阻塞事件循环。"""
        return await asyncio.to_thread(self.get_tuple, config)

    async def alist(self, config, *, filter=None, before=None, limit=None):
        """在线程中收集有限查询结果，再向异步调用方逐条返回。"""
        rows = await asyncio.to_thread(lambda: list(self.list(config, filter=filter, before=before, limit=limit)))
        for row in rows:
            yield row

    async def aput(self, config, checkpoint, metadata, new_versions):
        """异步保存完整快照，与同步实现共享同一事务规则。"""
        return await asyncio.to_thread(self.put, config, checkpoint, metadata, new_versions)

    async def aput_writes(self, config, writes, task_id, task_path=""):
        """异步保存节点待提交写入；数据库异常继续向图运行时传播。"""
        await asyncio.to_thread(self.put_writes, config, writes, task_id, task_path)

    async def adelete_thread(self, thread_id):
        """异步清理完整线程；调用方负责先确认归属与非活动状态。"""
        await asyncio.to_thread(self.delete_thread, thread_id)
