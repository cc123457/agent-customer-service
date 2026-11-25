"""独立Redis连接池：REDIS_URL管控制，FACT_CACHE_REDIS_URL管可淘汰事实缓存。"""
import os
import logging
from functools import lru_cache

from redis import Redis
from redis.backoff import NoBackoff
from redis.exceptions import RedisError
from redis.retry import Retry

from dotenv import load_dotenv

from utils.path_tool import get_abs_path

load_dotenv(get_abs_path(".env"), override=False)
logger = logging.getLogger(__name__)


@lru_cache(maxsize=8)
def _pooled_client(url):
    """同一URL复用线程安全连接池；不记录URL或凭据，不缓存一次连接失败的结果。"""
    return Redis.from_url(url, socket_connect_timeout=2, socket_timeout=2,
                          max_connections=64, retry_on_timeout=False, retry=Retry(NoBackoff(), 0))


def get_redis():
    """取得控制Redis供租约/限流使用；开发模式未连接时原有单进程兜底仍可用。"""
    url = os.getenv("REDIS_URL", "redis://127.0.0.1:6379/0")
    try:
        client = _pooled_client(url)
        client.ping()
        return client
    except (RedisError, ValueError):
        return None


def get_fact_cache_clients():
    """验证物理实例和内存策略；失败只禁用事实缓存，不混用控制实例。"""
    cache_url = os.getenv("FACT_CACHE_REDIS_URL", "").strip()
    if not cache_url:
        return None, None
    try:
        control = get_redis()
        if control is None:
            return None, None
        cache = _pooled_client(cache_url)
        # run_id可识别同一实例的不同DB编号、不同DNS别名，不能只比较URL字符串。
        if cache.info("server")["run_id"] == control.info("server")["run_id"]:
            logger.warning("事实缓存未启用：缓存和控制Redis必须是不同实例")
            return None, None
        cache_info, control_info = cache.info("memory"), control.info("memory")
        if (cache_info.get("maxmemory", 0) <= 0
                or cache_info.get("maxmemory_policy") not in ("allkeys-lfu", "allkeys-lru")
                or control_info.get("maxmemory", 0) <= 0 or control_info.get("maxmemory_policy") != "noeviction"):
            logger.warning("事实缓存未启用：需要独立内存预算、缓存LFU/LRU与控制noeviction策略")
            return None, None
        return cache, control
    except (RedisError, ValueError, KeyError):
        logger.warning("独立事实缓存暂不可用，保留SQL读取路径")
        return None, None
