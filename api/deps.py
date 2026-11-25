"""依赖组装：首次需要时创建聊天服务并按进程缓存，路由通过Depends取得共享组件。"""
from fastapi import Depends, HTTPException
from fastapi.security import HTTPAuthorizationCredentials, HTTPBearer
from functools import lru_cache
from sqlalchemy.exc import SQLAlchemyError
from redis.exceptions import RedisError
import os

from memory.session_lease import SessionLeaseStore
from services.chat_service import ChatService
from db.memory_factory import get_memory_repository
from utils.rate_limiter import RateLimiter
from utils.redis_client import get_redis
from utils.security import decode_token
from utils.config_handler import memory_conf

# 开发模式可使用单进程兜底；REDIS_CONTROL_REQUIRED=1时控制实例不可用应明确503。
redis_client = get_redis()

# 快速租约只管互斥；聊天消息、摘要与长期事实统一存入SQL。
session_leases = SessionLeaseStore(redis_client=redis_client)

# 限流器：每个用户每分钟最多 10 次请求
rate_limiter = RateLimiter(max_requests=10, window_seconds=60, redis_client=redis_client)

# HTTPBearer：从请求头 Authorization: Bearer <token> 中提取 token
security = HTTPBearer()


@lru_cache(maxsize=1)
def get_chat_service() -> ChatService:
    """供/chat取得唯一ChatService对象；缓存组件而非用户数据，每轮输入由open_turn另行接收。

    将共享 SessionLeaseStore、Agent 和 memory.yml 的策略参数组装在这里，
    使接口层不需要自己构造模型或配置存储；测试可替换这个依赖。
    """
    repository = get_history_repository()
    from agent.react_agent import ReactAgent
    from memory.checkpointer import SqlCheckpointSaver
    from memory.request_analyzer import MemoryRequestAnalyzer
    from utils.path_tool import get_abs_path
    from model.factory import chat_model
    semantic = None
    semantic_config = memory_conf.get("semantic", {})
    if semantic_config.get("enabled", False):
        from memory.semantic_index import SemanticMemoryIndex
        from model.factory import embed_model
        semantic = SemanticMemoryIndex(repository, get_abs_path(semantic_config["directory"]), embed_model,
                                       min_similarity=semantic_config.get("min_similarity", 0.3))
    return ChatService(
        ReactAgent(checkpointer=SqlCheckpointSaver(repository.engine)), session_leases, repository,
        request_analyzer=MemoryRequestAnalyzer(chat_model if memory_conf.get("analysis", {}).get("model_enabled", False) else None,
                                               timeout=memory_conf.get("analysis", {}).get("timeout", 3),
                                               input_budget=memory_conf.get("analysis", {}).get("input_budget", 4096),
                                               enabled=memory_conf.get("analysis", {}).get("enabled", True)),
        semantic_index=semantic,
        history_retriever=get_history_retriever(),
        **memory_conf["service"],
    )


@lru_cache(maxsize=1)
def get_history_retriever():
    """聊天与历史搜索API共用历史检索器；不为搜索初始化Agent或产品知识库。"""
    from memory.archive import MemoryArchive
    from memory.history_retrieval import HistoryRetriever
    from utils.path_tool import get_abs_path
    repository = get_history_repository()
    options = dict(memory_conf.get("history_retrieval", {}))
    semantic_options = dict(options.pop("semantic", {}))
    index = None
    if options.get("enabled", True) and semantic_options.pop("enabled", False):
        from memory.history_index import HistorySemanticIndex
        from model.factory import embed_model
        directory = get_abs_path(semantic_options.pop("directory"))
        index = HistorySemanticIndex(repository, directory, embed_model, **semantic_options)
    archive = MemoryArchive(repository, get_abs_path(memory_conf["retention"]["archive_directory"]))
    return HistoryRetriever(repository, semantic_index=index, archive=archive, **options)


def get_history_repository():
    """API侧数据库依赖：初始化失败返回503，不打印数据库凭据，不静默换另一份数据库。"""
    try:
        return get_memory_repository()
    except SQLAlchemyError as error:
        raise HTTPException(status_code=503, detail="历史数据库暂不可用") from error


def get_current_user(credentials: HTTPAuthorizationCredentials = Depends(security)) -> str:
    """鉴权依赖：解析 JWT，返回当前登录用户的 user_id；无效或过期则 401"""
    try:
        payload = decode_token(credentials.credentials)
    except Exception:
        raise HTTPException(status_code=401, detail="无效或过期的 token")

    user_id = payload.get("sub")
    if not user_id:
        raise HTTPException(status_code=401, detail="token 缺少用户信息")
    return user_id


def get_limited_user(user_id: str = Depends(get_current_user)) -> str:
    """鉴权 + 限流依赖：先解析用户，再检查限流；超限则 429"""
    if redis_client is None and os.getenv("REDIS_CONTROL_REQUIRED", "0") == "1":
        raise HTTPException(status_code=503, detail="请求控制服务暂不可用")
    try:
        allowed = rate_limiter.is_allowed(user_id)
    except RedisError as error:
        raise HTTPException(status_code=503, detail="请求控制服务暂不可用") from error
    if not allowed:
        raise HTTPException(status_code=429, detail="请求过于频繁，请稍后再试")
    return user_id
