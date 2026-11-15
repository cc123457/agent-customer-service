"""限流器：优先 Redis（ZSET 滑动窗口），无 Redis 时降级内存（固定窗口）。

is_allowed 接口保持一致。
"""
import time
from collections import defaultdict, deque


# 滑动窗口限流 Lua 脚本：删旧 + 计数 + 判断 + 记录，整体在 Redis 服务端原子执行，
# 消除 pipeline 无法覆盖的 check-then-act 竞态。
_RATE_LIMIT_LUA = """
local key = KEYS[1]
local window_start = tonumber(ARGV[1])
local max_requests = tonumber(ARGV[2])
local now = ARGV[3]
local ttl = tonumber(ARGV[4])
redis.call('ZREMRANGEBYSCORE', key, 0, window_start)
local count = redis.call('ZCARD', key)
if count >= max_requests then
    redis.call('EXPIRE', key, ttl)   -- 拒绝时也刷新过期（此时 key 已存在）
    return 0
end
redis.call('ZADD', key, now, now)
-- EXPIRE 必须在 ZADD 之后：首次请求时 key 尚不存在，提前 EXPIRE 会静默无效
redis.call('EXPIRE', key, ttl)
return 1
"""


class RateLimiter:
    def __init__(self, max_requests: int = 10, window_seconds: int = 60, redis_client=None):
        self.max_requests = max_requests
        self.window_seconds = window_seconds
        self._redis = redis_client
        self._hits = defaultdict(deque)   # 内存兜底：{key: deque[时间戳]}

    def is_allowed(self, key: str) -> bool:
        """判断某个 key 是否允许本次请求；允许则记录，超限返回 False"""
        if self._redis:
            return self._redis_is_allowed(key)
        return self._memory_is_allowed(key)

    def _memory_is_allowed(self, key: str) -> bool:
        """固定窗口（内存版）：踢掉窗口外老时间戳，计数超限则拒绝"""
        now = time.time()
        q = self._hits[key]
        while q and now - q[0] > self.window_seconds:
            q.popleft()
        if len(q) >= self.max_requests:
            return False
        q.append(now)
        return True

    def _redis_is_allowed(self, key: str) -> bool:
        """滑动窗口（Redis 版）：Lua 脚本整体原子执行，无 check-then-act 竞态"""
        rkey = f"ratelimit:{key}"
        now_ns = time.time_ns()
        window_start = now_ns - self.window_seconds * 1_000_000_000

        # 整个「删旧 + 计数 + 判断 + 记录 + 设过期」交给 Lua 脚本原子执行，返回 1 放行 / 0 拒绝
        allowed = self._redis.eval(
            _RATE_LIMIT_LUA,
            1,                  # 1 个 key
            rkey,               # KEYS[1]
            window_start,       # ARGV[1]
            self.max_requests,  # ARGV[2]
            now_ns,             # ARGV[3]
            self.window_seconds,  # ARGV[4] 过期时间 = 窗口长度（停止请求即自动删除）
        )
        return bool(allowed)
