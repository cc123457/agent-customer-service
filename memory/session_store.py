import json


class SessionStore:
    """会话记忆存储：优先 Redis（持久化、多实例共享），无 Redis 时降级内存。

    长度策略：append 只追加、不自动截断——压缩时机由上层（ChatService）通过
    extract_oldest 批量控制；这里只保留一个硬上限兜底，防止异常情况下无限增长。
    """

    def __init__(self, max_history: int = 20, redis_client=None, ttl_seconds: int = 1800):
        self.max_history = max_history   # 触发压缩的水位（由上层读取）
        self._redis = redis_client
        self.ttl_seconds = ttl_seconds   # 会话空闲过期时间（秒），默认 30 分钟
        self._store = {}          # 内存兜底（Redis 不可用时）
        self._store_summary = {}  # 内存兜底的对话摘要

    def get(self, session_id: str) -> list:
        """返回该会话的历史消息；不存在则返回空列表"""
        if self._redis:
            key = f"session:{session_id}"
            raw = self._redis.lrange(key, 0, -1)
            return [json.loads(x) for x in raw]
        return list(self._store.get(session_id, []))

    def append(self, session_id: str, role: str, content: str):
        """追加一条消息。不自动截断——压缩由上层 extract_oldest 触发，这里只做硬上限兜底。"""
        item = json.dumps({"role": role, "content": content}, ensure_ascii=False)
        if self._redis:
            key = f"session:{session_id}"
            self._redis.rpush(key, item)
            # 硬上限兜底：异常情况下（上层忘记压缩）防止无限增长
            if self._redis.llen(key) > self.max_history * 2:
                self._redis.ltrim(key, -self.max_history, -1)
            self._redis.expire(key, self.ttl_seconds)   # 每次写入刷新 TTL（滑动过期）
        else:
            messages = self._store.setdefault(session_id, [])
            messages.append({"role": role, "content": content})
            if len(messages) > self.max_history * 2:
                del messages[: len(messages) - self.max_history]

    def extract_oldest(self, session_id: str, count: int) -> list:
        """取出并删除最旧的 count 条消息（供压缩记忆批量使用），返回被取出的消息列表。"""
        if count <= 0:
            return []
        if self._redis:
            key = f"session:{session_id}"
            raw = self._redis.lrange(key, 0, count - 1)   # 先取出最旧 count 条
            self._redis.ltrim(key, count, -1)             # 再删除它们
            return [json.loads(x) for x in raw]
        else:
            messages = self._store.get(session_id, [])
            if not messages:
                return []
            oldest = messages[:count]
            del messages[:count]
            return oldest

    def get_summary(self, session_id: str) -> str:
        """返回该会话的对话摘要（压缩记忆）；无则返回空字符串"""
        if self._redis:
            raw = self._redis.get(f"session:{session_id}:summary")
            return raw.decode("utf-8") if raw else ""
        return self._store_summary.get(session_id, "")

    def set_summary(self, session_id: str, summary: str):
        """写入该会话的对话摘要；与消息一样带 TTL"""
        if self._redis:
            self._redis.set(f"session:{session_id}:summary", summary, ex=self.ttl_seconds)
        else:
            self._store_summary[session_id] = summary
