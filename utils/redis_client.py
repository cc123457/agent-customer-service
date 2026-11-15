"""Redis 连接（可选）：连接失败返回 None，调用方降级到内存实现"""
import os

from dotenv import load_dotenv

from utils.path_tool import get_abs_path

load_dotenv(get_abs_path(".env"), override=True)


def get_redis():
    """尝试连接 Redis，成功返回 client，失败返回 None（调用方降级）"""
    url = os.getenv("REDIS_URL", "redis://127.0.0.1:6379/0")
    try:
        import redis
        client = redis.from_url(url, socket_connect_timeout=2)
        client.ping()          # 探测连通性，失败抛异常
        return client
    except Exception:
        return None
