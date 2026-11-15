from memory.session_store import SessionStore


def test_memory_store_append_and_get():
    """内存版：追加后能取回，角色与内容正确"""
    store = SessionStore(max_history=10)
    store.append("s1", "user", "你好")
    store.append("s1", "assistant", "你好呀")
    msgs = store.get("s1")
    assert len(msgs) == 2
    assert msgs[0]["role"] == "user"
    assert msgs[1]["content"] == "你好呀"


def test_memory_store_truncate():
    """内存版：超过 max_history 时只保留最近 N 条"""
    store = SessionStore(max_history=2)
    for i in range(5):
        store.append("s1", "user", f"msg{i}")
    msgs = store.get("s1")
    assert len(msgs) == 2
    assert msgs[-1]["content"] == "msg4"


def test_redis_store_append_and_get(redis_client):
    """Redis 版：追加后能取回"""
    store = SessionStore(max_history=10, redis_client=redis_client)
    store.append("pytest_s1", "user", "你好")
    store.append("pytest_s1", "assistant", "你好呀")
    msgs = store.get("pytest_s1")
    assert len(msgs) == 2
    assert msgs[1]["content"] == "你好呀"


def test_redis_store_sets_ttl(redis_client):
    """Redis 版：写入后 key 会被设置过期时间（防止会话无限堆积）"""
    store = SessionStore(max_history=10, redis_client=redis_client)
    store.append("pytest_ttl", "user", "你好")
    ttl = redis_client.ttl("session:pytest_ttl")
    assert ttl > 0
