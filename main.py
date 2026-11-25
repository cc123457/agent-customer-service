from fastapi import FastAPI
from contextlib import asynccontextmanager

from api.auth import router as auth_router
from api.chat import router as chat_router
from api.memory import router as memory_router


@asynccontextmanager
async def lifespan(app):
    """服务退出时关闭已创建的语义索引，释放本地Chroma文件句柄；不为清理额外初始化服务。"""
    try:
        yield
    finally:
        from api.deps import get_chat_service, get_history_retriever
        if get_chat_service.cache_info().currsize:
            service = get_chat_service()
            if service.semantic_index:
                service.semantic_index.close()
        if get_history_retriever.cache_info().currsize:
            get_history_retriever().close()


app = FastAPI(title="智能客服 Agent API", version="1.0.0", lifespan=lifespan)

# 注册路由：所有具体接口都定义在 api/ 目录下，这里只负责挂载
app.include_router(auth_router)
app.include_router(chat_router)
app.include_router(memory_router)
