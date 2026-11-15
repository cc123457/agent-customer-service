"""依赖组装：在服务启动时创建单例，供路由层通过 Depends 注入"""
from fastapi import Depends, HTTPException
from fastapi.security import HTTPAuthorizationCredentials, HTTPBearer

from agent.react_agent import ReactAgent
from memory.session_store import SessionStore
from services.chat_service import ChatService
from utils.rate_limiter import RateLimiter
from utils.redis_client import get_redis
from utils.security import decode_token

# Redis 连接（可选）：失败为 None，会话/限流降级到内存
redis_client = get_redis()

# 全局单例：Agent 初始化代价高（加载模型、连接向量库），只建一次
agent = ReactAgent()
session_store = SessionStore(redis_client=redis_client)
chat_service = ChatService(agent, session_store)

# 限流器：每个用户每分钟最多 10 次请求
rate_limiter = RateLimiter(max_requests=10, window_seconds=60, redis_client=redis_client)

# HTTPBearer：从请求头 Authorization: Bearer <token> 中提取 token
security = HTTPBearer()


def get_chat_service() -> ChatService:
    """FastAPI 依赖注入入口：路由层通过 Depends 获取业务服务"""
    return chat_service


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
    if not rate_limiter.is_allowed(user_id):
        raise HTTPException(status_code=429, detail="请求过于频繁，请稍后再试")
    return user_id
