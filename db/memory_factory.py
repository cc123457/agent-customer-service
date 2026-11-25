"""部署配置与记忆数据库对象组装，不在导入模块时连接MySQL或创建文件。"""
import os
from functools import lru_cache
from threading import Lock

from sqlalchemy.engine import URL

from db.memory_repository import MemoryRepository
from utils.path_tool import get_abs_path


_factory_lock = Lock()


@lru_cache(maxsize=1)
def _build_repository(database_url):
    """在外层锁内初始化并缓存一个显式URL对应的仓库，避免首次并发重复建表。"""
    from utils.config_handler import memory_conf
    retention = memory_conf.get("retention", {})
    limits = {key: retention[key] for key in ("max_hot_bytes_per_user", "max_active_notes_per_user", "max_answer_bytes") if key in retention}
    cache_config = dict(memory_conf.get("fact_cache", {}))
    fact_cache = None
    if cache_config.pop("enabled", False):
        from memory.fact_cache import RedisFactCache
        from utils.redis_client import get_fact_cache_clients
        cache_client, directory_client = get_fact_cache_clients()
        fact_cache = RedisFactCache(cache_client, directory_client=directory_client,
                                   client_provider=get_fact_cache_clients, **cache_config)
    repository = MemoryRepository(database_url, limits=limits, fact_cache=fact_cache)
    repository.initialize()
    return repository


def get_memory_repository():
    """使用MEMORY_DATABASE_URL创建缓存仓库；未配置时显式使用独立本地memory.db。

    只增建记忆表，不迁移或清空现有agent.db；MySQL部署通过URL选择，不静默回退。
    """
    database_url = os.getenv("MEMORY_DATABASE_URL") or URL.create(
        "sqlite", database=get_abs_path("data/memory.db"),
    )
    with _factory_lock:
        return _build_repository(database_url)
