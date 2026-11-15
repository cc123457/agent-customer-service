"""pytest 配置：把项目根加入 sys.path，并提供 Redis fixture"""
import os
import sys

import pytest

# 项目根加入 sys.path，让测试能 import 项目模块
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from utils.redis_client import get_redis


@pytest.fixture
def redis_client():
    """Redis 客户端；不可用时跳过相关测试。

    teardown 阶段（yield 之后）清掉本测试用过的 key，
    保证测试可重复运行 —— 否则 Redis 里的残留数据会让断言数越跑越多。
    """
    client = get_redis()
    if client is None:
        pytest.skip("Redis 不可用，跳过 Redis 相关测试")

    yield client

    # 清理测试污染的 key（只删测试专用前缀，不动线上数据）
    for pattern in ("session:pytest_*", "ratelimit:pytest_*"):
        for key in client.scan_iter(match=pattern):
            client.delete(key)
