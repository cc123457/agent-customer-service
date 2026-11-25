"""会话租约只负责互斥与owner校验；消息、摘要和长期记忆统一由SQL保存。"""
import time
from threading import RLock

_RELEASE_LUA = """
if redis.call('GET', KEYS[1]) == ARGV[1] then return redis.call('DEL', KEYS[1]) end
return 0
"""


class SessionLeaseStore:
    """支持Redis共享租约；内存模式仅在单进程内共享，SQL仍保护持久化写入。"""

    def __init__(self, redis_client=None, clock=time.monotonic):
        """初始化租约字典与线程锁，不创建消息/摘要缓存，不发送Redis命令。"""
        self._redis = redis_client
        self._clock = clock
        self._leases = {}
        self._lock = RLock()

    def _purge_expired(self):
        """持有线程锁时清理过期内存租约，不影响SQL档案。"""
        now = self._clock()
        for scope, (_, deadline) in list(self._leases.items()):
            if deadline <= now:
                self._leases.pop(scope, None)

    def claim(self, session_id, owner, ttl_seconds):
        """由 open_turn 申请限时租约：成功 True，已有持有者则 False。

        owner 是每轮随机令牌；Redis 用 SET NX EX 支持多进程互斥，
        内存路径只保护当前进程。获取租约不追加消息、不刷新聊天数据 TTL。
        """
        if not session_id or not owner or ttl_seconds <= 0:
            raise ValueError("租约会话、owner和有效期必须有效")
        if self._redis is not None:
            return bool(self._redis.set(f"session:{session_id}:turn-lock", owner, nx=True, ex=ttl_seconds))
        with self._lock:
            self._purge_expired()
            if session_id in self._leases:
                return False
            self._leases[session_id] = (owner, self._clock() + ttl_seconds)
            return True

    def owns(self, session_id, owner):
        """返回owner是否仍持有快速租约，供诊断和测试使用。

        此检查不能代替写入授权；SQL写入在事务中独立检查数据库租约。
        """
        if self._redis is not None:
            current = self._redis.get(f"session:{session_id}:turn-lock")
            return current == owner or current == owner.encode("utf-8")
        with self._lock:
            self._purge_expired()
            return self._leases.get(session_id, (None,))[0] == owner

    def release(self, session_id, owner):
        """由 close_turn 清理租约：仅 owner 匹配才删除并返回 True，否则 False。

        Redis 用 _RELEASE_LUA 将比较与删除一起执行，防止旧请求删新锁。
        不删除消息或摘要；Redis 故障仍抛异常。
        """
        if self._redis is not None:
            return bool(self._redis.eval(_RELEASE_LUA, 1, f"session:{session_id}:turn-lock", owner))
        with self._lock:
            if self._leases.get(session_id, (None,))[0] != owner:
                return False
            self._leases.pop(session_id)
            return True
