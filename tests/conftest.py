"""pytest 配置：把项目根加入 sys.path，并提供 Redis fixture"""
import os
import sys

import pytest

# 项目根加入 sys.path，让测试能 import 项目模块
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import redis
from db.memory_repository import MemoryRepository
from db.memory_schema import metadata


@pytest.fixture
def redis_client():
    """Redis 客户端；不可用时跳过相关测试。

    teardown 阶段（yield 之后）清掉本测试用过的 key，
    保证测试可重复运行 —— 否则 Redis 里的残留数据会让断言数越跑越多。
    """
    test_url = os.getenv("TEST_REDIS_URL", "redis://127.0.0.1:6391/15")
    client = redis.from_url(test_url, socket_connect_timeout=1, socket_timeout=2)
    try:
        client.ping()
    except redis.RedisError:
        if os.getenv("REQUIRE_REDIS") == "1":
            pytest.fail("REQUIRE_REDIS=1，但测试 Redis 不可用")
        pytest.skip("测试 Redis 不可用")

    try:
        yield client
    finally:
        for pattern in ("session:pytest_*", "ratelimit:pytest_*"):
            for key in client.scan_iter(match=pattern):
                client.delete(key)
        client.close()


@pytest.fixture(params=["sqlite", "mysql"])
def repository(request, tmp_path):
    """两个真实SQL后端运行同一批断言；MySQL库名必须以agent_memory_test开头。"""
    if request.param == "sqlite":
        url = "sqlite:///" + (tmp_path / "memory.db").as_posix()
    else:
        url = os.getenv("TEST_MEMORY_DATABASE_URL")
        if not url:
            if os.getenv("REQUIRE_MYSQL") == "1":
                pytest.fail("REQUIRE_MYSQL=1但缺少测试数据库URL")
            pytest.skip("未配置测试MySQL")
        from sqlalchemy.engine import make_url
        if not (make_url(url).database or "").startswith("agent_memory_test"):
            pytest.fail("拒绝清理非测试数据库")
    repo = MemoryRepository(url)
    repo.initialize()
    try:
        yield repo
    finally:
        if request.param == "mysql":
            with repo.engine.begin() as connection:
                for table in reversed(metadata.sorted_tables):
                    connection.execute(table.delete())
        repo.engine.dispose()


@pytest.fixture
def fact_cache_redis(redis_client):
    """连接独立可淘汰测试实例；不修改服务配置，不清空整个Redis。"""
    client = redis.from_url(os.getenv("TEST_FACT_CACHE_REDIS_URL", "redis://127.0.0.1:6392/15"),
                            socket_connect_timeout=1, socket_timeout=2)
    try:
        client.ping()
        assert client.info("server")["run_id"] != redis_client.info("server")["run_id"], "缓存测试必须使用另一Redis实例"
    except redis.RedisError:
        client.close()
        if os.getenv("REQUIRE_FACT_CACHE_REDIS") == "1":
            pytest.fail("REQUIRE_FACT_CACHE_REDIS=1，但独立缓存Redis不可用")
        pytest.skip("独立缓存测试Redis不可用")
    try:
        yield client
    finally:
        client.close()
