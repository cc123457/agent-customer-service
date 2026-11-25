"""历史问答独立Chroma集合：按轮次分块，索引只提供候选，原文由SQL与冷档案复核。"""
import hashlib
import json
from pathlib import Path
from threading import Lock


def turn_fingerprint(turn):
    """绑定轮次、消息ID、角色和正文，防止旧向量代表另一版历史原文。"""
    source = [turn["turn_id"], [[row["id"], row["role"], row["content"]] for row in turn["messages"]]]
    return hashlib.sha256(json.dumps(source, ensure_ascii=False).encode("utf-8")).hexdigest()


class HistorySemanticIndex:
    """仅索引当前SQL授权的候选窗口；不同数据库、用户和长期事实集合相互隔离。"""

    def __init__(self, repository, directory, embeddings, candidate_limit=12,
                 min_similarity=0.3, chunk_chars=1000, chunk_overlap=150):
        """创建独立持久集合；Embedding在按用户分条带的同步锁内，不持有全局索引锁。"""
        import chromadb
        from model.embedding_space import embedding_space
        if candidate_limit < 1 or not 0 <= min_similarity <= 1 or not 0 <= chunk_overlap < chunk_chars:
            raise ValueError("历史向量索引配置无效")
        self.embeddings = embeddings
        # 复用数据库隔离指纹，与Redis是否可用无关。
        self.partition = repository._cache_partition
        Path(directory).mkdir(parents=True, exist_ok=True)
        self.client = chromadb.PersistentClient(path=str(directory))
        self.collection = self.client.get_or_create_collection(
            "conversation_history_" + embedding_space(embeddings), metadata={"hnsw:space": "cosine"}, embedding_function=None,
        )
        self.candidate_limit = candidate_limit
        self.min_similarity = min_similarity
        self.chunk_chars = chunk_chars
        self.chunk_overlap = chunk_overlap
        self._sync_locks = tuple(Lock() for _ in range(32))

    def _where(self, user_id, turns):
        """查询前限定数据库、用户及SQL已过滤的轮次，避免全库召回后才过滤。"""
        return {"$and": [{"partition": self.partition}, {"user_id": user_id},
                          {"turn_id": {"$in": [turn["turn_id"] for turn in turns]}}]}

    def _chunks(self, user_id, turns):
        """按字符窗口分块双方原文；保留整轮指纹，命中任一块后恢复完整问答。"""
        chunks = []
        for turn in turns:
            fingerprint = turn_fingerprint(turn)
            text = "\n".join(row["role"] + ": " + row["content"] for row in turn["messages"])
            for position in range(0, len(text), self.chunk_chars - self.chunk_overlap):
                content = text[position:position + self.chunk_chars]
                identity = hashlib.sha256(
                    f"{self.partition}:{user_id}:{turn['turn_id']}:{fingerprint}:{position}".encode("utf-8")
                ).hexdigest()
                chunks.append((identity, content, {
                    "partition": self.partition, "user_id": user_id, "turn_id": turn["turn_id"],
                    "session_id": turn["session_id"], "fingerprint": fingerprint,
                    "created_at": turn["created_at"],
                }))
                if position + self.chunk_chars >= len(text):
                    break
        return chunks

    def _sync(self, user_id, turns):
        """只补充变化的块，分批生成向量；不删除候选窗口以外的其他轮次。"""
        slot = int(hashlib.sha256(user_id.encode("utf-8")).hexdigest()[:8], 16) % len(self._sync_locks)
        with self._sync_locks[slot]:
            chunks = self._chunks(user_id, turns)
            old = self.collection.get(where=self._where(user_id, turns), include=["metadatas"])
            existing = set(old["ids"])
            wanted = {row[0] for row in chunks}
            changed = [row for row in chunks if row[0] not in existing]
            for offset in range(0, len(changed), 16):
                batch = changed[offset:offset + 16]
                vectors = self.embeddings.embed_documents([row[1] for row in batch])
                if len(vectors) != len(batch):
                    raise ValueError("历史Embedding返回数量不匹配")
                self.collection.upsert(ids=[row[0] for row in batch], embeddings=vectors,
                                       metadatas=[row[2] for row in batch])
            stale = existing - wanted
            if stale:
                self.collection.delete(ids=list(stale))
            return len(chunks)

    def search(self, user_id, query, turns):
        """先同步授权候选，再余弦召回并按轮次取最高分；不返回索引中的正文。"""
        if not turns:
            return []
        chunk_count = self._sync(user_id, turns)
        vector = self.embeddings.embed_query(query)
        result = self.collection.query(query_embeddings=[vector], where=self._where(user_id, turns),
                                       n_results=min(self.candidate_limit, chunk_count), include=["metadatas", "distances"])
        best = {}
        for meta, distance in zip(result["metadatas"][0], result["distances"][0]):
            score = 1 - distance
            identity = meta["turn_id"]
            if score >= self.min_similarity and (identity not in best or score > best[identity]["score"]):
                best[identity] = {"turn_id": identity, "fingerprint": meta["fingerprint"], "score": score}
        return sorted(best.values(), key=lambda row: row["score"], reverse=True)

    def remove_turns(self, user_id, turns):
        """清理本次候选中已被遗忘策略排除的旧向量，不跨用户/数据库删除。"""
        if turns:
            self.collection.delete(where=self._where(user_id, turns))

    def close(self):
        """服务退出时释放本地Chroma句柄；调用方不应与正在执行的检索同时关闭。"""
        self.client.close()
