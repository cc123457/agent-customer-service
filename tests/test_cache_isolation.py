"""独立真实Redis验证滑动续期、目录上限、发布竞态与控制实例保护。"""
import json
import os
import time
from concurrent.futures import ThreadPoolExecutor
from threading import Event
from unittest.mock import Mock
from uuid import uuid4

import pytest
from fastapi import HTTPException
from redis.exceptions import ConnectionError as RedisConnectionError

from memory.fact_cache import RedisFactCache, _PUBLISH_LUA


def fact(value="X10", expires_at=None):
    """返回有限字段事实，作为SQL加载器的输入数据；Redis行为使用真实服务。"""
    result = dict(id="fact", user_id="owner", memory_key="device_model", value=value,
                  status="active", source_message_id=1)
    if expires_at is not None:
        result["expires_at"] = expires_at
    return result


@pytest.fixture
def managed_cache(redis_client, fact_cache_redis):
    """每例使用独立目录/正文前缀，清理仅扫描本例前缀，不修改服务配置。"""
    prefix = "memory:pytest_isolated:" + uuid4().hex
    cache = RedisFactCache(fact_cache_redis, directory_client=redis_client, key_prefix=prefix, ttl_jitter_seconds=0)
    yield cache
    for client in (redis_client, fact_cache_redis):
        for key in client.scan_iter(match=prefix + ":*"):
            client.delete(key)


def read(cache, version=1, limit=20, loader=None):
    """固定查询范围，便于比较真实TTL和加载器次数。"""
    return cache.get_or_load("db", "owner", version, ["device_model"], limit, loader or (lambda: [fact()]))


def test_valid_hit_only_refreshes_below_threshold(managed_cache):
    cache = managed_cache
    loader = Mock(return_value=[fact()])
    read(cache, loader=loader)
    key = cache.cache_key("db", "owner", 1, ["device_model"], 20)
    assert 7_199_000 <= cache._redis.pttl(key) <= 7_200_000
    read(cache, loader=loader)
    assert cache.stats("owner")["refreshes"] == 0
    cache._redis.pexpire(key, 500)
    assert read(cache, loader=loader)[0]["value"] == "X10"
    assert cache._redis.pttl(key) > 7_199_000
    assert cache.stats("owner")["refreshes"] == 1
    loader.assert_called_once()


def test_empty_result_has_five_minutes_and_does_not_slide(managed_cache):
    cache = managed_cache
    loader = Mock(return_value=[])
    assert read(cache, loader=loader) == []
    key = cache.cache_key("db", "owner", 1, ["device_model"], 20)
    assert 299_000 <= cache._redis.pttl(key) <= 300_000
    cache._redis.pexpire(key, 500)
    assert read(cache, loader=loader) == [] and cache._redis.pttl(key) <= 500
    assert cache.stats("owner")["refreshes"] == 0
    loader.assert_called_once()


def test_sliding_cannot_extend_business_expiration(managed_cache):
    cache = managed_cache
    deadline = time.time() + 5
    row = fact(expires_at=deadline)
    loader = Mock(return_value=[row])
    read(cache, loader=loader)
    key = cache.cache_key("db", "owner", 1, ["device_model"], 20)
    assert 0 < cache._redis.pttl(key) <= 5000
    read(cache, loader=loader)
    assert cache._redis.pttl(key) <= max(1, int((deadline - time.time()) * 1000) + 5)
    cache._wall_clock = lambda: deadline + 1
    assert read(cache, loader=loader) == []
    assert loader.call_count == 2 and cache.stats("owner")["invalid_entries"] == 1


def test_user_key_count_is_bounded_and_old_shape_reloads(managed_cache):
    cache = managed_cache
    cache._directory.max_keys = 2
    for size in range(1, 6):
        assert read(cache, limit=size)[0]["value"] == "X10"
    keys = [cache.cache_key("db", "owner", 1, ["device_model"], size) for size in range(1, 6)]
    assert cache._redis.exists(*keys) == 2
    assert cache._directory_client.hlen(cache._directory.key("db", "owner")) == 3
    assert cache.stats("owner")["reclaimed_keys"] == 3
    assert not cache._redis.exists(keys[0])
    assert read(cache, limit=1)[0]["value"] == "X10"


def test_correcting_during_publication_cannot_restore_old_key(managed_cache, monkeypatch):
    cache = managed_cache
    published, release = Event(), Event()
    original = cache._directory.confirm
    def pause(partition, user_id, version, key, token):
        """暂停已写正文但未确认的旧发布者，构造明确的跨实例交错。"""
        if version == 1:
            published.set()
            assert release.wait(5)
        return original(partition, user_id, version, key, token)
    monkeypatch.setattr(cache._directory, "confirm", pause)
    with ThreadPoolExecutor(max_workers=1) as pool:
        pending = pool.submit(read, cache)
        try:
            assert published.wait(5)
            cache.note_invalidation("owner", partition="db", version=2)
            assert read(cache, version=2, loader=lambda: [fact("X20")])[0]["value"] == "X20"
        finally:
            release.set()
        assert pending.result(timeout=5)[0]["value"] == "X10"
    assert not cache._redis.exists(cache.cache_key("db", "owner", 1, ["device_model"], 20))
    assert read(cache, version=2)[0]["value"] == "X20"


def test_publication_with_expired_lease_never_creates_a_body(managed_cache):
    cache = managed_cache
    key = cache.cache_key("db", "owner", 1, ["device_model"], 20)
    result = cache._redis.eval(_PUBLISH_LUA, 1, key, "set", b"payload", 7200000, 0, 0, 3600000)
    assert result == 0 and not cache._redis.exists(key)


def test_directory_failure_does_not_trust_untracked_warm_body(managed_cache, monkeypatch):
    cache = managed_cache
    read(cache)
    monkeypatch.setattr(cache._directory_client, "eval", Mock(side_effect=RedisConnectionError("directory unavailable")))
    assert read(cache, loader=lambda: [fact("X20")])[0]["value"] == "X20"
    assert cache.stats("owner")["bypasses"] == 1


def test_startup_cache_outage_can_revalidate_and_recover(redis_client, fact_cache_redis):
    clock = Mock(return_value=100)
    provider = Mock(side_effect=[(None, None), (fact_cache_redis, redis_client)])
    prefix = "memory:pytest_recovery:" + uuid4().hex
    cache = RedisFactCache(client_provider=provider, clock=clock, key_prefix=prefix, ttl_jitter_seconds=0)
    try:
        assert not cache.stats("owner")["available"]
        provider.assert_not_called()
        read(cache)
        clock.return_value = 106
        read(cache)
        read(cache)
        assert cache.stats("owner")["hits"] == 1 and provider.call_count == 2
    finally:
        for client in (redis_client, fact_cache_redis):
            for key in client.scan_iter(match=prefix + ":*"):
                client.delete(key)


@pytest.mark.parametrize("same_instance,cache_limit,cache_policy,control_policy,expected", [
    (False, 1024, "allkeys-lfu", "noeviction", True),
    (True, 1024, "allkeys-lfu", "noeviction", False),
    (False, 0, "allkeys-lfu", "noeviction", False),
    (False, 1024, "noeviction", "noeviction", False),
    (False, 1024, "allkeys-lfu", "allkeys-lfu", False),
])
def test_factory_checks_physical_instance_and_memory_policy(monkeypatch, same_instance, cache_limit, cache_policy, control_policy, expected):
    import utils.redis_client as clients
    control = Mock()
    cache = Mock()
    control.info.side_effect = lambda section: {"run_id": "control"} if section == "server" else {"maxmemory": 1024, "maxmemory_policy": control_policy}
    cache.info.side_effect = lambda section: {"run_id": "control" if same_instance else "cache"} if section == "server" else {"maxmemory": cache_limit, "maxmemory_policy": cache_policy}
    monkeypatch.setenv("FACT_CACHE_REDIS_URL", "redis://private:do-not-log@cache/1")
    monkeypatch.setattr(clients, "get_redis", lambda: control)
    monkeypatch.setattr(clients, "_pooled_client", lambda url: cache)
    actual = clients.get_fact_cache_clients()
    assert actual == ((cache, control) if expected else (None, None))


def test_unconfigured_cache_never_falls_back_to_control(monkeypatch):
    import utils.redis_client as clients
    monkeypatch.delenv("FACT_CACHE_REDIS_URL", raising=False)
    pool = Mock()
    monkeypatch.setattr(clients, "_pooled_client", pool)
    assert clients.get_fact_cache_clients() == (None, None)
    pool.assert_not_called()


def test_factory_reuses_validated_separate_real_pools(monkeypatch, redis_client, fact_cache_redis):
    import utils.redis_client as clients
    monkeypatch.setenv("REDIS_URL", os.getenv("TEST_REDIS_URL", "redis://127.0.0.1:6391/15"))
    monkeypatch.setenv("FACT_CACHE_REDIS_URL", os.getenv("TEST_FACT_CACHE_REDIS_URL", "redis://127.0.0.1:6392/15"))
    first = clients.get_fact_cache_clients()
    second = clients.get_fact_cache_clients()
    assert first[0] is second[0] and first[1] is second[1] and first[0] is not first[1]
    assert first[0].info("server")["run_id"] != first[1].info("server")["run_id"]


def test_control_failure_returns_503_instead_of_silently_resetting_global_limit(monkeypatch):
    from api import deps
    monkeypatch.setattr(deps, "redis_client", None)
    monkeypatch.setenv("REDIS_CONTROL_REQUIRED", "1")
    with pytest.raises(HTTPException) as caught:
        deps.get_limited_user("owner")
    assert caught.value.status_code == 503
    monkeypatch.setattr(deps, "redis_client", Mock())
    monkeypatch.setattr(deps.rate_limiter, "is_allowed", Mock(side_effect=RedisConnectionError("lost")))
    with pytest.raises(HTTPException) as caught:
        deps.get_limited_user("owner")
    assert caught.value.status_code == 503


def test_cache_memory_eviction_does_not_remove_control_lock_or_rate_state(managed_cache, redis_client, fact_cache_redis):
    """只在明确专用的小缓存实例施压；不修改maxmemory，不触碰6379或其他用户键。"""
    if os.getenv("TEST_FACT_CACHE_DEDICATED") != "1":
        pytest.skip("仅在显式声明的独立Redis测试实例运行内存施压")
    from memory.session_lease import SessionLeaseStore
    from utils.rate_limiter import RateLimiter
    cache = managed_cache
    scope = "pytest_isolated_" + uuid4().hex
    leases = SessionLeaseStore(redis_client)
    limiter = RateLimiter(1, 60, redis_client)
    before = fact_cache_redis.info("stats")["evicted_keys"]
    assert leases.claim(scope, "owner", 60) and limiter.is_allowed(scope)
    try:
        for number in range(2048):
            fact_cache_redis.set(cache._key_prefix + ":pressure:" + str(number), b"x" * 16384, ex=30)
            if number % 64 == 0 and fact_cache_redis.info("stats")["evicted_keys"] > before:
                break
        assert fact_cache_redis.info("stats")["evicted_keys"] > before
        assert leases.owns(scope, "owner") and not limiter.is_allowed(scope)
        assert redis_client.ttl("session:" + scope + ":turn-lock") > 0
    finally:
        leases.release(scope, "owner")
        redis_client.delete("ratelimit:" + scope)
