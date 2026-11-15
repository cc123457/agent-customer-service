from utils.rate_limiter import RateLimiter


def test_memory_rate_limiter():
    """内存版：阈值内放行，超限拒绝"""
    rl = RateLimiter(max_requests=3, window_seconds=60)  # 无 redis → 内存版
    results = [rl.is_allowed("user1") for _ in range(5)]
    assert results == [True, True, True, False, False]


def test_redis_rate_limiter(redis_client):
    """Redis 版（Lua 滑动窗口）：阈值内放行，超限拒绝"""
    rl = RateLimiter(max_requests=3, window_seconds=60, redis_client=redis_client)
    results = [rl.is_allowed("pytest_user") for _ in range(5)]
    assert results == [True, True, True, False, False]


def test_redis_rate_limiter_sets_ttl(redis_client):
    """Redis 版：限流 key 会被设置过期时间（停止请求后自动删除）"""
    rl = RateLimiter(max_requests=3, window_seconds=60, redis_client=redis_client)
    rl.is_allowed("pytest_ttl")
    ttl = redis_client.ttl("ratelimit:pytest_ttl")
    assert ttl > 0
