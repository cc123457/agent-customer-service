"""热正文转为可校验的本地gzip冷档案；保留消息ID/来源引用，不把摘要成功当作删除授权。"""
import gzip
import hashlib
import json
import os
from pathlib import Path
from uuid import uuid4

from sqlalchemy import and_, delete, insert, or_, select, update

from db.memory_extensions import archived_messages, archive_batches, retired_archives
from db.memory_schema import conversations, messages, summary_versions, turns
from db.checkpoint_schema import checkpoints, checkpoint_writes


class MemoryArchive:
    """先导出并校验，再短事务换成冷档案索引；数据库失败会留下可清理孤立文件，不丢原文。"""

    def __init__(self, repository, directory):
        """目录由服务端配置，不接受客户端文件路径；默认本地目录不是异地备份。"""
        self.repository = repository
        self.directory = Path(directory).resolve()

    def _path(self, file_name):
        """只允许归档目录内的普通文件名，拒绝目录穿越或任意外部路径。"""
        if Path(file_name).name != file_name or (self.directory / file_name).resolve().parent != self.directory:
            raise ValueError("归档文件名无效")
        return self.directory / file_name

    def _load(self, record):
        """每次恢复前验证完整压缩文件SHA256，损坏或缺失明确报错，不返回伪造空正文。"""
        blob = self._path(record["file_name"]).read_bytes()
        if hashlib.sha256(blob).hexdigest() != record["sha256"]:
            raise ValueError("归档校验失败")
        rows = [json.loads(line) for line in gzip.decompress(blob).decode("utf-8").splitlines()]
        if len(rows) != record["message_count"]:
            raise ValueError("归档条数不一致")
        return {row["id"]: row["content"] for row in rows}

    def hydrate(self, rows):
        """给已经授权且分页后的历史行恢复冷正文；只读取这些消息关联的归档，不扫描所有文件。"""
        if not rows:
            return rows
        with self.repository.engine.connect() as connection:
            mappings = connection.execute(select(archived_messages.c.message_id, archive_batches, retired_archives.c.retired_at).join(
                archive_batches, archive_batches.c.id == archived_messages.c.batch_id,
            ).outerjoin(retired_archives, retired_archives.c.batch_id == archive_batches.c.id)
                .where(archived_messages.c.message_id.in_([row["id"] for row in rows]))).mappings().all()
        by_id = {row["message_id"]: row for row in mappings}
        loaded = {}
        result = []
        for row in rows:
            restored = dict(row)
            if row["id"] in by_id:
                batch = by_id[row["id"]]
                if batch["retired_at"] is not None:
                    restored.update(content="[原文已超过存档保留期限并清理]", archived=True, expired=True)
                    result.append(restored)
                    continue
                if batch["id"] not in loaded:
                    loaded[batch["id"]] = self._load(batch)
                restored["content"] = loaded[batch["id"]][row["id"]]
                restored["archived"] = True
            result.append(restored)
        return result

    def archive(self, cutoff, *, limit=100, apply=False):
        """只选择已摘要覆盖或失败的过期原文；默认预览，apply时验证文件和原行未变后清空热正文。"""
        repo = self.repository
        with repo.engine.connect() as connection:
            rows = [dict(row) for row in connection.execute(select(messages).join(
                conversations, conversations.c.scope == messages.c.scope,
            ).join(turns, turns.c.id == messages.c.turn_id).outerjoin(
                archived_messages, archived_messages.c.message_id == messages.c.id,
            ).where(messages.c.created_at < cutoff, messages.c.content != "",
                    archived_messages.c.message_id.is_(None), conversations.c.lease_until <= repo.clock(),
                    or_(turns.c.status == "failed", and_(turns.c.status == "completed", messages.c.id <= conversations.c.summary_upto)))
                .order_by(messages.c.id).limit(min(max(limit, 1), 1000))).mappings()]
        report = {"eligible_messages": len(rows), "hot_bytes": sum(len(row["content"].encode()) for row in rows), "applied": False}
        if not apply or not rows:
            return report
        identity = uuid4().hex
        file_name = identity + ".jsonl.gz"
        self.directory.mkdir(parents=True, exist_ok=True)
        payload = "\n".join(json.dumps({"id": row["id"], "content": row["content"]}, ensure_ascii=False) for row in rows).encode()
        blob = gzip.compress(payload, mtime=0)
        temporary = self._path(identity + ".tmp")
        with temporary.open("xb") as handle:
            handle.write(blob)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, self._path(file_name))
        batch = dict(id=identity, file_name=file_name, sha256=hashlib.sha256(blob).hexdigest(),
                     message_count=len(rows), created_at=repo.clock())
        recovered = self._load(batch)
        if recovered != {row["id"]: row["content"] for row in rows}:
            raise ValueError("导出校验不一致，未修改SQL原文")
        with repo.engine.begin() as connection:
            for scope in sorted({row["scope"] for row in rows}):
                current = connection.execute(select(conversations).where(conversations.c.scope == scope).with_for_update()).mappings().one()
                if current["lease_until"] > repo.clock():
                    raise RuntimeError("归档时会话已重新活动，未裁剪原文")
            connection.execute(insert(archive_batches).values(**batch))
            for row in rows:
                current = connection.execute(select(messages.c.content).where(messages.c.id == row["id"])).scalar_one()
                if current != row["content"]:
                    raise RuntimeError("归档期间原文发生变化")
                connection.execute(insert(archived_messages).values(message_id=row["id"], batch_id=identity))
                connection.execute(update(messages).where(messages.c.id == row["id"]).values(content=""))
        return dict(report, applied=True, batch_id=identity)

    def retire_archives(self, cutoff, apply=False, limit=100):
        """冷档案到期先事务标记失效，再删除本目录对应文件；失败可重试，保留来源ID及过期标记。"""
        with self.repository.engine.begin() as connection:
            rows = [dict(row) for row in connection.execute(select(archive_batches).outerjoin(
                retired_archives, retired_archives.c.batch_id == archive_batches.c.id,
            ).where(
                archive_batches.c.created_at < cutoff, retired_archives.c.batch_id.is_(None),
            ).order_by(archive_batches.c.created_at).limit(limit)).mappings()]
            if apply:
                from memory.checkpointer import upsert
                for row in rows:
                    upsert(connection, retired_archives, dict(batch_id=row["id"], retired_at=self.repository.clock()), replace=False)
        if apply:
            # 事务标记成功后若文件删除失败，下一次只重试仍存在的已失效文件，不阻塞新的到期批次。
            existing = [path.name for path in self.directory.glob("*.jsonl.gz")]
            with self.repository.engine.connect() as connection:
                cleanup = connection.execute(select(archive_batches).join(retired_archives,
                    retired_archives.c.batch_id == archive_batches.c.id).where(archive_batches.c.file_name.in_(existing))).mappings().all()
            for row in cleanup:
                path = self._path(row["file_name"])
                if path.exists():
                    path.unlink()
        return {"expired_archive_files": len(rows), "applied": apply}

    def prune_versions(self, keep=5, apply=False):
        """保留每会话最新若干摘要版本；当前摘要仍在会话表，不受历史版本清理影响。"""
        if keep < 1:
            raise ValueError("至少保留一版摘要")
        count = 0
        with self.repository.engine.begin() as connection:
            scopes = connection.execute(select(summary_versions.c.scope).distinct()).scalars().all()
            for scope in scopes:
                ids = connection.execute(select(summary_versions.c.id).where(summary_versions.c.scope == scope)
                                         .order_by(summary_versions.c.id.desc()).offset(keep)).scalars().all()
                count += len(ids)
                if apply and ids:
                    connection.execute(delete(summary_versions).where(summary_versions.c.id.in_(ids)))
        return {"obsolete_summary_versions": count, "applied": apply}

    def prune_checkpoints(self, completed_cutoff, failed_cutoff, apply=False, limit=100):
        """按终态请求整线程清理，不删除活动线程或单个祖先，避免破坏DeltaChannel恢复父链。"""
        count = 0
        with self.repository.engine.begin() as connection:
            rows = connection.execute(select(turns).join(conversations, conversations.c.scope == turns.c.scope).where(
                conversations.c.lease_until <= self.repository.clock(),
                or_(and_(turns.c.status == "completed", turns.c.created_at < completed_cutoff),
                    and_(turns.c.status == "failed", turns.c.created_at < failed_cutoff)),
            ).order_by(turns.c.created_at).limit(limit).with_for_update()).mappings().all()
            for row in rows:
                exists = connection.execute(select(checkpoints.c.checkpoint_id).where(checkpoints.c.thread_id == row["id"]).limit(1)).first()
                if exists:
                    count += 1
                    if apply:
                        connection.execute(delete(checkpoint_writes).where(checkpoint_writes.c.thread_id == row["id"]))
                        connection.execute(delete(checkpoints).where(checkpoints.c.thread_id == row["id"]))
        return {"expired_checkpoint_threads": count, "applied": apply}
