import json
import asyncio
import logging

from fastapi import APIRouter, Depends, HTTPException
from fastapi.responses import StreamingResponse
from pydantic import BaseModel, Field, field_validator
from redis.exceptions import RedisError
from sqlalchemy.exc import SQLAlchemyError
from db.memory_repository import DuplicateRequestError

from api.deps import get_chat_service, get_limited_user
from services.chat_service import ChatService
from services.chat_types import ContextBudgetError, SessionBusyError
from services.chat_types import SessionCapacityError, SessionLeaseError
from memory.history_retrieval import HistoryArchiveError

router = APIRouter()
logger = logging.getLogger(__name__)


class ChatRequest(BaseModel):
    """请求体：用户输入的一句话 + 会话 ID"""
    message: str = Field(min_length=1, max_length=4000)
    session_id: str = Field(min_length=1, max_length=128, pattern=r"^[A-Za-z0-9_-]+$")
    request_id: str | None = Field(default=None, min_length=1, max_length=64, pattern=r"^[A-Za-z0-9_-]+$")

    @field_validator("message")
    @classmethod
    def nonblank_message(cls, value):
        """由 Pydantic 在解析请求时调用，输入 message 字符串并原样返回。

        全为空白则抛 ValueError，由 FastAPI 转成 422；正常消息不在这里裁剪。
        """
        if not value.strip():
            raise ValueError("消息不能为空白")
        return value


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
    # 1. 归档本轮用户输入，准备预算内上下文；不把完整档案直接传给模型。
    try:
        if req.request_id:
            turn = await service.open_turn(req.session_id, req.message, user_id, request_id=req.request_id)
        else:
            turn = await service.open_turn(req.session_id, req.message, user_id)
    except (SessionBusyError, SessionLeaseError) as error:
        raise HTTPException(status_code=409, detail=str(error)) from error
    except DuplicateRequestError as error:
        raise HTTPException(status_code=409, detail=str(error)) from error
    except (ContextBudgetError, SessionCapacityError) as error:
        raise HTTPException(status_code=413, detail=str(error)) from error
    except TimeoutError as error:
        raise HTTPException(status_code=504, detail="上下文准备超时，请稍后重试") from error
    except RedisError as error:
        raise HTTPException(status_code=503, detail="会话存储暂不可用，请稍后重试") from error
    except HistoryArchiveError as error:
        raise HTTPException(status_code=503, detail="历史冷档案暂不可读取，请稍后重试") from error
    except SQLAlchemyError as error:
        raise HTTPException(status_code=503, detail="历史数据库暂不可用，请稍后重试") from error

    return stream_response(service, turn, user_id)


def stream_response(service, turn, user_id):
    """新请求与恢复请求共用SSE收尾；错误带外部request_id，客户端据此恢复，不泄露内部线程ID。"""
    async def event_stream():
        """StreamingResponse 消费的异步生成器，捕获外层 turn 和认证 user_id。

        调用 reply_turn 并将事件字典序列化为 SSE；流中普通错误发送 error，
        取消向上传播，finally 释放租约；连接仍可用且清理成功时发送 DONE。
        此处负责传输与错误展示，不负责模型决策或直接操作 Redis。
        """
        try:
            async for item in service.reply_turn(turn, user_id):
                yield f"data: {json.dumps(item, ensure_ascii=False)}\n\n"
        except asyncio.CancelledError:
            raise
        except Exception:
            logger.exception("聊天执行失败")
            item = {"type": "error", "content": "本轮处理或回答保存失败，请稍后重试；未删除已有历史。",
                    "request_id": getattr(turn, "request_id", None),
                    "resumable": bool(getattr(service.agent, "checkpointer", None))}
            yield f"data: {json.dumps(item, ensure_ascii=False)}\n\n"
        finally:
            await service.close_turn(turn)
        yield "data: [DONE]\n\n"

    return StreamingResponse(event_stream(), media_type="text/event-stream")
