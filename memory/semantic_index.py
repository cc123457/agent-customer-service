"""个人记忆独立Chroma集合；SQL是事实依据，索引只返回候选ID，纠正/删除后强制SQL复核。"""
import hashlib
from pathlib import Path
from threading import RLock


class SemanticMemoryIndex:
    """显式启用才调用Embedding；每个用户单独同步，检索先过滤用户，再检查当前SQL版本。"""

    def __init__(self, repository, directory, embeddings, min_similarity=0.3, candidate_limit=12):
        """使用独立路径和集合，不修改产品知识库；可注入可控向量验证索引管理逻辑。"""
        import chromadb
        from model.embedding_space import embedding_space
        self.repository = repository
        self.embeddings = embeddings
        self.directory = Path(directory)
        self.directory.mkdir(parents=True, exist_ok=True)
        self.client = chromadb.PersistentClient(path=str(self.directory))
        self.collection = self.client.get_or_create_collection(
            "personal_memory_" + embedding_space(embeddings), metadata={"hnsw:space": "cosine"}, embedding_function=None,
        )
        self.min_similarity = min_similarity
        self.candidate_limit = candidate_limit
        self.lock = RLock()

    def close(self):
        """关闭Chroma公开客户端，释放Windows文件句柄；重复关闭安全，临时评测目录才能正常清理。"""
        self.client.close()

    @staticmethod
    def fingerprint(row):
        """内容及来源共同生成版本指纹，防止更正后的SQL事实使用过期向量得分。"""
        value = str(row["source_message_id"]) + "\n" + row["memory_key"] + "\n" + row["value"]
        return hashlib.sha256(value.encode("utf-8")).hexdigest()

    def sync_user(self, user_id):
        """按SQL有效条目增量更新本人索引并删除失效ID；失败不改变SQL事实。"""
        with self.lock:
            rows = self.repository.active_memories(user_id)
            current = self.collection.get(where={"user_id": user_id}, include=["metadatas"])
            old = dict(zip(current["ids"], current["metadatas"]))
            wanted = {row["id"] for row in rows}
            stale = [identity for identity in old if identity not in wanted]
            if stale:
                self.collection.delete(ids=stale)
            changed = [row for row in rows if old.get(row["id"], {}).get("fingerprint") != self.fingerprint(row)]
            if changed:
                texts = [row.get("entity_id", "profile") + " " + row["memory_key"] + ": " + row["value"] for row in changed]
                vectors = self.embeddings.embed_documents(texts)
                self.collection.upsert(ids=[row["id"] for row in changed], embeddings=vectors,
                                       documents=texts, metadatas=[{"user_id": user_id, "fingerprint": self.fingerprint(row)} for row in changed])

    def search(self, user_id, query):
        """只接受本人向量候选，SQL复核active状态和指纹；返回SQL正文，不返回索引缓存正文。"""
        with self.lock:
            if not self.collection.get(where={"user_id": user_id}, limit=1)["ids"]:
                return []
            result = self.collection.query(query_embeddings=[self.embeddings.embed_query(query)],
                                           where={"user_id": user_id}, n_results=self.candidate_limit,
                                           include=["metadatas", "distances"])
            candidates = {
                identity: (meta["fingerprint"], 1 - distance)
                for identity, meta, distance in zip(result["ids"][0], result["metadatas"][0], result["distances"][0])
                if 1 - distance >= self.min_similarity
            }
            valid = self.repository.active_memories(user_id, list(candidates))
            rows = [dict(row, semantic_score=candidates[row["id"]][1]) for row in valid
                    if self.fingerprint(row) == candidates[row["id"]][0]]
            return sorted(rows, key=lambda row: row["semantic_score"], reverse=True)
