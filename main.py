from fastapi import FastAPI

from api.auth import router as auth_router
from api.chat import router as chat_router


app = FastAPI(title="智能客服 Agent API", version="1.0.0")

# 注册路由：所有具体接口都定义在 api/ 目录下，这里只负责挂载
app.include_router(auth_router)
app.include_router(chat_router)
