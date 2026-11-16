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


def test_memory_store_extract_oldest():
    """内存版：extract_oldest 取出并删除最旧的 N 条（供压缩记忆批量使用）"""
    store = SessionStore(max_history=10)
    for i in range(6):
        store.append("s1", "user", f"msg{i}")
    oldest = store.extract_oldest("s1", 4)
    assert len(oldest) == 4
    assert oldest[0]["content"] == "msg0"
    assert oldest[-1]["content"] == "msg3"
    # 取出后应只剩 msg4、msg5
    remain = store.get("s1")
    assert [m["content"] for m in remain] == ["msg4", "msg5"]


def test_memory_store_hard_cap():
    """内存版：异常情况下（上层未压缩）超过硬上限时兜底截断，防止无限增长"""
    store = SessionStore(max_history=2)   # 硬上限 = max_history * 2 = 4
    for i in range(5):
        store.append("s1", "user", f"msg{i}")
    msgs = store.get("s1")
    assert len(msgs) == 2
    assert msgs[-1]["content"] == "msg4"


def test_memory_store_summary():
    """内存版：摘要可存取，未写入时返回空字符串"""
    store = SessionStore(max_history=10)
    assert store.get_summary("s1") == ""
    store.set_summary("s1", "之前聊过故障排除")
    assert store.get_summary("s1") == "之前聊过故障排除"


def test_redis_store_append_and_get(redis_client):
    """Redis 版：追加后能取回"""
    store = SessionStore(max_history=10, redis_client=redis_client)
    store.append("pytest_s1", "user", "你好")
    store.append("pytest_s1", "assistant", "你好呀")
    msgs = store.get("pytest_s1")
    assert len(msgs) == 2
    assert msgs[1]["content"] == "你好呀"


def test_redis_store_extract_oldest(redis_client):
    """Redis 版：extract_oldest 取出并删除最旧的 N 条"""
    store = SessionStore(max_history=10, redis_client=redis_client)
    for i in range(4):
        store.append("pytest_ex", "user", f"msg{i}")
    oldest = store.extract_oldest("pytest_ex", 2)
    assert [m["content"] for m in oldest] == ["msg0", "msg1"]
    remain = store.get("pytest_ex")
    assert [m["content"] for m in remain] == ["msg2", "msg3"]


def test_redis_store_sets_ttl(redis_client):
    """Redis 版：写入后 key 会被设置过期时间（防止会话无限堆积）"""
    store = SessionStore(max_history=10, redis_client=redis_client)
    store.append("pytest_ttl", "user", "你好")
    ttl = redis_client.ttl("session:pytest_ttl")
    assert ttl > 0
