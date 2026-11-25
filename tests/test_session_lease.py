"""快速租约真实Redis/内存双路径，不再测试已删除的Redis消息和摘要实现。"""
from unittest.mock import Mock

import pytest
from memory.session_lease import SessionLeaseStore


@pytest.fixture(params=["memory", "redis"])
def leases(request):
    """为两个租约后端复用相同互斥和owner断言。"""
    client = request.getfixturevalue("redis_client") if request.param == "redis" else None
    return SessionLeaseStore(redis_client=client)


def test_one_owner_per_session_and_different_sessions_independent(leases):
    assert leases.claim("pytest_lease_a", "first", 120)
    assert not leases.claim("pytest_lease_a", "second", 120)
    assert leases.claim("pytest_lease_b", "second", 120)
    assert leases.owns("pytest_lease_a", "first")
    assert not leases.release("pytest_lease_a", "wrong-owner")
    assert leases.release("pytest_lease_a", "first")
    assert leases.claim("pytest_lease_a", "next", 120)
    assert not leases.release("pytest_lease_a", "first")
    assert leases.owns("pytest_lease_a", "next")
    leases.release("pytest_lease_a", "next")
    leases.release("pytest_lease_b", "second")


def test_expired_memory_owner_cannot_release_new_owner():
    clock = Mock(return_value=100)
    leases = SessionLeaseStore(clock=clock)
    assert leases.claim("same", "old", 10)
    clock.return_value = 111
    assert not leases.owns("same", "old")
    assert leases.claim("same", "new", 10)
    assert not leases.release("same", "old")
    assert leases.owns("same", "new")


def test_expired_redis_owner_cannot_release_new_owner(redis_client):
    leases = SessionLeaseStore(redis_client=redis_client)
    assert leases.claim("pytest_lease_expire", "old", 120)
    redis_client.pexpire("session:pytest_lease_expire:turn-lock", 0)
    assert leases.claim("pytest_lease_expire", "new", 120)
    assert not leases.release("pytest_lease_expire", "old")
    assert leases.owns("pytest_lease_expire", "new")
    leases.release("pytest_lease_expire", "new")
