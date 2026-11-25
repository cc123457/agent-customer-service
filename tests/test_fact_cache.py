"""真实SQL和独立Redis验证版本、隔离和并发；仅故障路径注入Redis异常。"""
import asyncio
import json
import time
from concurrent.futures import ThreadPoolExecutor
from threading import Event, Lock
from unittest.mock import Mock
from uuid import uuid4

import httpx
import pytest
from fastapi import FastAPI
from redis.exceptions import ConnectionError as RedisConnectionError
from sqlalchemy import event, select
from sqlalchemy.exc import SQLAlchemyError

from db.memory_repository import MemoryRepository
from db.memory_schema import memories, memory_cache_versions
from memory.fact_cache import RedisFactCache
from memory.long_term import MemoryCandidate, extract_memories, recall_memories
from services.chat_service import ChatService


@pytest.fixture
def cache(redis_client, fact_cache_redis):
    """真实独立Redis运行Lua、TTL与目录；不以Python重写Lua模拟原子行为。"""
    prefix = "memory:pytest_facts:" + uuid4().hex
    instance = RedisFactCache(fact_cache_redis, directory_client=redis_client,
                              key_prefix=prefix, ttl_jitter_seconds=0)
    yield instance
    for client in (redis_client, fact_cache_redis):
        for key in client.scan_iter(match=prefix + ":*"):
            client.delete(key)


def save_input(repository, text, user="1001", candidates=None):
    """通过真实事务写入原文和事实；不调用模型，返回原文ID。"""
    session = uuid4().hex
    scope = ChatService.session_scope(user, session)
    owner = uuid4().hex
    assert repository.claim(scope, user, session, owner, 120)
    try:
        turn, message_id = repository.start_request(
            scope, user, owner, uuid4().hex, text,
            extract_memories(text) if candidates is None else candidates,
        )
        repository.finish_request(scope, owner, turn, "已收到")
        return message_id
    finally:
        repository.release(scope, owner)


def find_device(repository, user="1001"):
    """固定查询形状，以便比较缓存与原SQL字段召回结果。"""
    return repository.search_memories(user, keys=["device_model"])


def device_key(repository, cache, user="1001", version=None):
    """定位本测试的设备查询键，不扫描其他用户或数据库的缓存。"""
    return cache.cache_key(repository._cache_partition, user,
                           repository.memory_version(user) if version is None else version,
                           ["device_model"], 20)


def test_hit_skips_fact_query_but_keeps_sql_version_check(repository, cache):
    """第二次查询应省掉事实正文SQL，而不是伪称整个请求完全不访问数据库。"""
    repository.fact_cache = cache
    save_input(repository, "我用的是X10")
    statements = []

    def record_sql(connection, cursor, statement, parameters, context, executemany):
        """只统计真实执行的SELECT，不替换数据库返回值。"""
        if statement.lstrip().upper().startswith("SELECT"):
            statements.append(statement)

    event.listen(repository.engine, "before_cursor_execute", record_sql)
    try:
        first = find_device(repository)
        statements.clear()
        second = find_device(repository)
    finally:
        event.remove(repository.engine, "before_cursor_execute", record_sql)
    assert first == second and first[0]["value"] == "X10"
    assert "memory_cache_versions" in statements[0]
    assert all("FROM user_memories" not in statement for statement in statements)
    assert any("memory_forget_rules" in statement for statement in statements)
    stats = cache.stats("1001")
    assert (stats["hits"], stats["misses"], stats["sql_loads"], stats["hit_rate"]) == (1, 1, 1, 0.5)
    assert cache.ttl_seconds - 60 <= cache._redis.ttl(device_key(repository, cache)) <= cache.ttl_seconds


def test_version_invalidates_correction_conflict_and_both_deletes(repository, cache):
    """更正、歧义、澄清、API删除和用户显式遗忘都不能继续命中旧事实。"""
    repository.fact_cache = cache
    save_input(repository, "我用的是X10")
    assert find_device(repository)[0]["value"] == "X10"
    old_key = device_key(repository, cache)
    save_input(repository, "实际型号是X20")
    assert find_device(repository)[0]["value"] == "X20"
    assert not cache._redis.exists(old_key)  # SQL提交后按目录主动清理旧版本。
    save_input(repository, "我的设备型号是Q1。我的设备型号是Q2。")
    assert find_device(repository) == []
    found = recall_memories(repository, "1001", "我的设备型号是什么？", [])
    assert any(row.get("needs_clarification") for row in found)
    save_input(repository, "实际型号是B8")
    row = find_device(repository)[0]
    assert repository.delete_memory("1001", row["id"])
    assert find_device(repository) == []
    save_input(repository, "我用的是X30")
    assert find_device(repository)[0]["value"] == "X30"
    save_input(repository, "忘记我的设备型号")
    assert find_device(repository) == []


def test_users_queries_and_databases_do_not_share_cached_results(repository, cache, tmp_path):
    """相同字段和版本在不同身份/数据库下不能互串，查询数量也属于缓存键。"""
    repository.fact_cache = cache
    save_input(repository, "我用的是X10", "1001")
    save_input(repository, "我用的是B8", "1002")
    assert find_device(repository, "1001")[0]["value"] == "X10"
    assert find_device(repository, "1002")[0]["value"] == "B8"
    assert device_key(repository, cache, "1001") != device_key(repository, cache, "1002")
    assert repository.search_memories("1001", keys=["floor_type"]) == []
    assert len(repository.search_memories("1001", keys=["device_model"], limit=1)) == 1
    other = MemoryRepository("sqlite:///" + (tmp_path / "other.db").as_posix(), fact_cache=cache)
    other.initialize()
    try:
        save_input(other, "我用的是Z9")
        assert find_device(other)[0]["value"] == "Z9"
        assert find_device(repository)[0]["value"] == "X10"
    finally:
        other.engine.dispose()


def test_empty_cache_expiry_and_corrupt_entries_reload_sql(repository, cache):
    """空结果也缓存；真实Redis到期、坏JSON和串用户内容均重新查询SQL。"""
    repository.fact_cache = cache
    assert find_device(repository) == []
    assert find_device(repository) == []
    key = device_key(repository, cache)
    assert cache.stats("1001")["hits"] == 1
    cache._redis.pexpire(key, 0)
    assert find_device(repository) == []
    cache._redis.set(key, b"broken-json")
    assert find_device(repository) == []
    entry = json.loads(cache._redis.get(key))
    entry["user_id"] = "1002"
    cache._redis.set(key, json.dumps(entry))
    assert find_device(repository) == []
    assert cache.stats("1001")["invalid_entries"] == 2


def test_other_repository_observes_sql_version_without_redis_invalidation(repository, cache):
    """另一服务未接Redis也能更新事实；读取者根据SQL版本绕过旧缓存。"""
    repository.fact_cache = cache
    save_input(repository, "我用的是X10")
    assert find_device(repository)[0]["value"] == "X10"
    writer = MemoryRepository(repository.engine.url)
    try:
        save_input(writer, "实际型号是X20")
        assert find_device(repository)[0]["value"] == "X20"
    finally:
        writer.engine.dispose()


def test_sql_failure_rolls_back_facts_and_version(repository, cache):
    """候选批次中途失败时，正式事实和版本必须一起回滚，原缓存仍然正确。"""
    repository.fact_cache = cache
    save_input(repository, "我用的是X10")
    assert find_device(repository)[0]["value"] == "X10"
    before = repository.memory_version("1001")
    with pytest.raises(ValueError, match="用户原文"):
        save_input(repository, "实际型号是X20", candidates=[
            MemoryCandidate("device_model", "X20"), MemoryCandidate("floor_type", "瓷砖"),
        ])
    assert repository.memory_version("1001") == before
    assert find_device(repository)[0]["value"] == "X10"
    assert cache.stats("1001")["hits"] == 1


def test_late_old_reader_cannot_repopulate_current_version(repository, cache, monkeypatch):
    """暂停一个读到旧值的请求，更新完成后再让它回填，验证新请求仍只能取新值。"""
    repository.fact_cache = cache
    save_input(repository, "我用的是X10")
    old_version = repository.memory_version("1001")
    loaded, proceed = Event(), Event()
    original = cache.get_or_load

    def delayed(partition, user_id, version, keys, limit, loader):
        """只暂停旧版本加载，不替换SQL数据或Redis。"""
        def load_then_pause():
            """保留真实SQL快照，给并发写入留出确定的测试时间窗口。"""
            rows = loader()
            loaded.set()
            assert proceed.wait(5)
            return rows
        return original(partition, user_id, version, keys, limit,
                        load_then_pause if version == old_version else loader)

    monkeypatch.setattr(cache, "get_or_load", delayed)
    with ThreadPoolExecutor(max_workers=1) as executor:
        old_read = executor.submit(find_device, repository)
        try:
            assert loaded.wait(5)
            save_input(repository, "实际型号是X20")
        finally:
            proceed.set()
        assert old_read.result(timeout=5)[0]["value"] == "X10"
    assert find_device(repository)[0]["value"] == "X20"
    assert not cache._redis.exists(device_key(repository, cache, version=old_version))


def test_same_process_cold_reads_share_one_fill(cache):
    """同一进程多个冷读只调用一次加载器，后续读取真实Redis中的同一条结果。"""
    started, proceed = Event(), Event()
    counter_lock = Lock()
    count = 0

    def load():
        """可控加载器只统计回填次数，不模拟Redis内部并发语义。"""
        nonlocal count
        with counter_lock:
            count += 1
        started.set()
        assert proceed.wait(5)
        return []

    with ThreadPoolExecutor(max_workers=8) as executor:
        futures = [executor.submit(cache.get_or_load, "one-db", "1001", 0,
                                   ["device_model"], 20, load) for _ in range(8)]
        try:
            assert started.wait(5)
        finally:
            proceed.set()
        assert all(future.result(timeout=5) == [] for future in futures)
    assert count == 1
    assert (cache.stats("1001")["hits"], cache.stats("1001")["misses"]) == (7, 1)


def test_redis_failure_bypasses_with_cooldown_and_recovers(repository, cache):
    """仅注入Redis连接故障；SQL查询、更正和版本更新都使用真实数据库。"""
    client = Mock()
    client.eval.side_effect = RedisConnectionError("do not log connection secrets")
    clock = Mock(return_value=100)
    cache._redis, cache._clock = client, clock
    repository.fact_cache = cache
    save_input(repository, "我用的是X10")
    assert find_device(repository)[0]["value"] == "X10"
    save_input(repository, "实际型号是X20")
    assert find_device(repository)[0]["value"] == "X20"
    assert client.eval.call_count == 1
    stats = cache.stats("1001")
    assert stats["bypasses"] == 2 and stats["errors"] == 1 and stats["hit_rate"] is None
    client.eval.side_effect = None
    client.eval.side_effect = [[], 1]
    clock.return_value = 106
    assert find_device(repository)[0]["value"] == "X20"
    assert client.eval.call_count == 3


def test_cache_write_failure_and_sql_loader_error_have_different_behavior(repository, cache):
    """Redis写失败返回已查得的真实事实；SQL失败必须传播，不能假装没有记忆。"""
    client = Mock()
    client.eval.side_effect = [[], RedisConnectionError("unavailable")]
    cache._redis = client
    repository.fact_cache = cache
    save_input(repository, "我用的是X10")
    assert find_device(repository)[0]["value"] == "X10"
    assert cache.stats("1001")["errors"] == 1
    fresh = RedisFactCache(Mock(eval=Mock(return_value=[])), directory_client=cache._directory_client,
                           key_prefix=cache._key_prefix + ":sql")
    loader = Mock(side_effect=SQLAlchemyError("database unavailable"))
    with pytest.raises(SQLAlchemyError):
        fresh.get_or_load("db", "1001", 1, ["device_model"], 20, loader)
    assert fresh._redis.eval.call_count == 1


def test_oversized_entry_is_not_cached_and_stats_are_user_scoped(repository, cache):
    """超限正文只通过SQL返回，不写缓存；不同用户看不到彼此统计。"""
    repository.fact_cache = cache
    cache.max_payload_bytes = 32
    save_input(repository, "我用的是X10")
    assert find_device(repository)[0]["value"] == "X10"
    assert find_device(repository)[0]["value"] == "X10"
    assert not cache._redis.exists(device_key(repository, cache))
    assert cache.stats("1001")["oversized"] == 2
    assert cache.stats("1002")["sql_loads"] == 0
    assert cache.stats("1002")["hit_rate"] is None


def test_notes_and_unknown_fields_keep_original_sql_path(repository, cache):
    """笔记仍实时检索，缓存不会改变关键词排序或把未知字段偷偷纳入缓存。"""
    repository.fact_cache = cache
    save_input(repository, "我用的是X10")
    save_input(repository, "记住：我对噪声敏感。")
    rows = repository.search_memories("1001", terms=["噪声"], keys=["device_model"])
    assert any(row["value"] == "X10" for row in rows)
    assert any("噪声" in row["value"] for row in rows)
    rows_again = repository.search_memories("1001", terms=["噪声"], keys=["device_model"])
    assert rows == rows_again
    before = cache.stats("1001")
    assert repository.search_memories("1001", keys=["unknown_field"]) == []
    assert cache.stats("1001") == before


def test_cache_stats_api_authentication_and_identity(repository, cache):
    """API要求JWT，只读本人统计，并且取得统计不会额外算成一次缓存命中。"""
    from api.deps import get_history_repository
    from api.memory import router
    from utils.security import create_token
    repository.fact_cache = cache
    save_input(repository, "我用的是X10")
    find_device(repository)
    find_device(repository)
    app = FastAPI()
    app.include_router(router)
    app.dependency_overrides[get_history_repository] = lambda: repository

    async def run():
        """通过真实ASGI路由和JWT鉴权检查接口返回。"""
        async with httpx.AsyncClient(transport=httpx.ASGITransport(app=app), base_url="http://test") as client:
            response = await client.get("/memories/cache-stats")
            assert response.status_code in (401, 403)
            mine = await client.get("/memories/cache-stats", headers={"Authorization": "Bearer " + create_token("1001")})
            other = await client.get("/memories/cache-stats", headers={"Authorization": "Bearer " + create_token("1002")})
            assert mine.status_code == other.status_code == 200
            assert mine.json()["hit_rate"] == 0.5 and mine.json()["scope"] == "current_process"
            assert other.json()["hits"] == 0 and other.json()["hit_rate"] is None
            assert "X10" not in mine.text
    asyncio.run(run())


def test_chat_service_consumes_cache_and_correction_in_new_sessions(repository, cache):
    """完整聊天服务跨会话复用事实缓存，更正后新上下文必须使用新型号。"""
    from tests.memory_helpers import complete, service
    repository.fact_cache = cache
    save_input(repository, "我用的是X10")
    chat = service(repository)

    async def run():
        """Agent只提供受控回答，用户归档、缓存、版本与上下文组装均运行真实代码。"""
        first = await complete(chat, "first", "我的机器人型号是什么？")
        second = await complete(chat, "second", "机器人充电要注意什么？")
        assert "X10" in str(first.messages) and "X10" in str(second.messages)
        assert cache.stats("1001")["hits"] >= 1
        save_input(repository, "实际型号是X20")
        corrected = await complete(chat, "third", "我的机器人型号是什么？")
        assert "X20" in str(corrected.messages) and "X10" not in str(corrected.messages)
    asyncio.run(run())


def test_disabled_and_unconfigured_cache_use_sql(repository):
    """未注入缓存保持原路径；启用但Redis不可用时明确统计旁路。"""
    save_input(repository, "我用的是X10")
    assert find_device(repository)[0]["value"] == "X10"
    assert repository.memory_cache_stats("1001")["enabled"] is False
    repository.fact_cache = RedisFactCache()
    assert find_device(repository)[0]["value"] == "X10"
    stats = repository.memory_cache_stats("1001")
    assert stats["enabled"] and not stats["available"] and stats["bypasses"] == 1


@pytest.mark.parametrize("enabled", [True, False])
def test_repository_factory_wires_cache_configuration(tmp_path, monkeypatch, enabled):
    """生产装配读取配置开关与TTL；测试只替换Redis连接，不初始化真实Agent。"""
    from db.memory_factory import _build_repository
    from utils.config_handler import memory_conf
    import utils.redis_client
    monkeypatch.setitem(memory_conf, "fact_cache", {"enabled": enabled, "ttl_seconds": 42})
    monkeypatch.setattr(utils.redis_client, "get_fact_cache_clients", lambda: (Mock(), Mock()))
    repository = _build_repository.__wrapped__("sqlite:///" + (tmp_path / "factory.db").as_posix())
    try:
        assert (repository.fact_cache is not None) == enabled
        if enabled:
            assert repository.fact_cache.ttl_seconds == 42
            assert repository.memory_cache_stats("1001")["available"]
    finally:
        repository.engine.dispose()


def test_existing_database_can_add_version_table_without_altering_facts(repository):
    """新增表可用于升级旧库；初始化不修改已有事实，版本0缓存仍会被后续写入推进。"""
    save_input(repository, "我用的是X10")
    with repository.engine.connect() as connection:
        before = [dict(row) for row in connection.execute(select(memories)).mappings()]
    memory_cache_versions.drop(repository.engine)
    repository.initialize()
    assert repository.memory_version("1001") == 0
    with repository.engine.connect() as connection:
        assert [dict(row) for row in connection.execute(select(memories)).mappings()] == before
    save_input(repository, "实际型号是X20")
    assert repository.memory_version("1001") == 1
    assert find_device(repository)[0]["value"] == "X20"
