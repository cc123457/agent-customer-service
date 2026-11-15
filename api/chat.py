import json

from fastapi import APIRouter, Depends
from fastapi.responses import StreamingResponse
from pydantic import BaseModel

from api.deps import get_chat_service, get_limited_user
from services.chat_service import ChatService

router = APIRouter()


class ChatRequest(BaseModel):
    """请求体：用户输入的一句话 + 会话 ID"""
    message: str
    session_id: str = "default"   # 缺省使用默认会话，前端可传自定义 ID 区分不同用户


@router.get("/health")
def health():
    """健康检查接口：确认服务还活着"""
    return {"status": "ok"}


@router.post("/chat")
async def chat(
    req: ChatRequest,
    service: ChatService = Depends(get_chat_service),
    user_id: str = Depends(get_limited_user),
):
    """对话接口：以 SSE 流式返回 Agent 的回答（异步链路，支持 MCP 异步工具）。

    响应为 SSE 事件流（text/event-stream），逐行推送，格式：
    - data: {"type": "action", "content": "调用工具 xxx，参数 ..."}   # Agent 动作轨迹
    - data: {"type": "answer", "content": "..."}                     # 最终回答内容
    - data: [DONE]                                                   # 流结束标记

    需登录（Authorization: Bearer <token>），并受每分钟限流约束。
    """
    # 1. 存入本轮用户输入，取出完整历史
    messages = service.build_messages(req.session_id, req.message)

    async def event_stream():
        reply_parts = []
        # 2. 异步流式产出结构化输出（action=动作轨迹，answer=回答）
        async for item in service.stream_reply(messages, user_id):
            # 只有 answer 才存入会话记忆，action 不存
            if item["type"] == "answer":
                reply_parts.append(item["content"])
            yield f"data: {json.dumps(item, ensure_ascii=False)}\n\n"
        # 3. 流结束后写回完整回答
        service.save_reply(req.session_id, "".join(reply_parts))
        yield "data: [DONE]\n\n"

    return StreamingResponse(event_stream(), media_type="text/event-stream")
