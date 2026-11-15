import json


class SessionStore:
    """会话记忆存储：优先 Redis（持久化、多实例共享），无 Redis 时降级内存。

    get / append 接口保持不变，调用方无感知。
    """

    def __init__(self, max_history: int = 20, redis_client=None, ttl_seconds: int = 1800):
        self.max_history = max_history
        self._redis = redis_client
        self.ttl_seconds = ttl_seconds   # 会话空闲过期时间（秒），默认 30 分钟
        self._store = {}          # 内存兜底（Redis 不可用时）

    def get(self, session_id: str) -> list:
        """返回该会话的历史消息；不存在则返回空列表"""
        if self._redis:
            key = f"session:{session_id}"
            raw = self._redis.lrange(key, 0, -1)
            return [json.loads(x) for x in raw]
        return list(self._store.get(session_id, []))

    def append(self, session_id: str, role: str, content: str):
        """往会话追加一条消息；超出上限时丢弃最旧的消息"""
        item = json.dumps({"role": role, "content": content}, ensure_ascii=False)
        if self._redis:
            key = f"session:{session_id}"
            self._redis.rpush(key, item)
            self._redis.ltrim(key, -self.max_history, -1)   # 只保留最近 max_history 条
            self._redis.expire(key, self.ttl_seconds)        # 每次写入刷新 TTL（滑动过期）
        else:
            messages = self._store.setdefault(session_id, [])
            messages.append({"role": role, "content": content})
            if len(messages) > self.max_history:
                del messages[: len(messages) - self.max_history]
