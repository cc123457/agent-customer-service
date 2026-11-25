"""按认证身份查看归档历史和长期记忆；客户端不能指定另一个user_id。"""
import asyncio
from datetime import datetime

from fastapi import APIRouter, Depends, HTTPException, Query

from api.deps import get_chat_service, get_current_user, get_history_repository, get_limited_user, get_history_retriever
from api.chat import stream_response
from services.chat_types import SessionBusyError
from db.memory_repository import DuplicateRequestError
from services.chat_service import ChatService
from sqlalchemy.exc import SQLAlchemyError
from memory.history_retrieval import BUSINESS_TZ, parse_history_scope, HistoryArchiveError

router = APIRouter()


@router.get("/history/search")
async def search_history(query: str = Query(min_length=1, max_length=4000),
                         user_id: str = Depends(get_limited_user), retriever=Depends(get_history_retriever),
                         session_id: str | None = Query(None, min_length=1, max_length=128, pattern=r"^[A-Za-z0-9_-]+$"),
                         after: datetime | None = None, before: datetime | None = None):
    """搜索本人历史完整问答；显式会话/时间优先于问题中的日期，时间上界不包含。

    不带时区的API时间按UTC+8解释；认证身份不可由query参数替换。
    语义服务失败可退BM25，SQL或冷档案损坏明确503，不伪造“没有历史”。
    """
    if not query.strip():
        raise HTTPException(422, "历史检索问题不能为空白")
    if not retriever.enabled:
        raise HTTPException(503, "历史检索未启用")
    try:
        plan = parse_history_scope(query, session_id, retriever.repository.clock())
        start = (after.replace(tzinfo=BUSINESS_TZ) if after.tzinfo is None else after).timestamp() if after else plan.after
        end = (before.replace(tzinfo=BUSINESS_TZ) if before.tzinfo is None else before).timestamp() if before else plan.before
        found = await asyncio.to_thread(retriever.search, user_id, query,
                                        session_id=session_id or plan.session_id, after=start, before=end)
        return {"turns": found}
    except (SQLAlchemyError, HistoryArchiveError) as error:
        raise HTTPException(503, "历史原文暂不可读取，请稍后重试") from error
    except ValueError as error:
        raise HTTPException(422, str(error)) from error


@router.get("/conversations/{session_id}/messages")
def history(session_id: str, limit: int = Query(50, ge=1, le=100), before_id: int | None = Query(None, ge=1),
            user_id: str = Depends(get_current_user), repository=Depends(get_history_repository)):
    """返回本人会话的原文分页与请求状态；before_id用于向前翻页，不读取其他用户的数据。"""
    scope = ChatService.session_scope(user_id, session_id)
    rows = repository.history(scope, user_id, limit, before_id)
    from memory.archive import MemoryArchive
    from utils.config_handler import memory_conf
    from utils.path_tool import get_abs_path
    try:
        rows = MemoryArchive(repository, get_abs_path(memory_conf["retention"]["archive_directory"])).hydrate(rows)
    except (OSError, ValueError, KeyError) as error:
        raise HTTPException(503, "历史冷档案暂不可读取，请稍后重试") from error
    return {"messages": rows, "next_before_id": rows[-1]["id"] if len(rows) == limit else None}


@router.get("/memories")
def memories(user_id: str = Depends(get_current_user), repository=Depends(get_history_repository)):
    """返回当前用户有效长期记忆及来源ID，不返回删除状态的记忆或其他用户的条目。"""
    return {"memories": repository.list_memories(user_id)}


@router.delete("/memories/{memory_id}")
def delete_memory(memory_id: str, user_id: str = Depends(get_current_user), repository=Depends(get_history_repository)):
    """按本人身份软删除记忆；不存在与越权均404，保留原始档案/审计事件，不再用于长期召回。"""
    if not repository.delete_memory(user_id, memory_id):
        raise HTTPException(status_code=404, detail="未找到有效记忆")
    return {"deleted": True}


@router.get("/memories/cache-stats")
def memory_cache_stats(user_id: str = Depends(get_current_user), repository=Depends(get_history_repository)):
    """返回本人当前进程的事实缓存统计；多进程不聚合，不包含其他用户数据或记忆正文。"""
    return repository.memory_cache_stats(user_id)


@router.get("/conversations/{session_id}/requests/{request_id}/checkpoints")
async def checkpoints(session_id: str, request_id: str, user_id: str = Depends(get_current_user),
                      service=Depends(get_chat_service)):
    """只查看本人请求的节点检查点概况，原始图状态不会通过此接口泄露。"""
    try:
        return {"checkpoints": await service.list_checkpoints(session_id, request_id, user_id)}
    except LookupError as error:
        raise HTTPException(404, str(error)) from error
    except SQLAlchemyError as error:
        raise HTTPException(503, "检查点数据库暂不可用") from error


@router.post("/conversations/{session_id}/requests/{request_id}/resume")
async def resume(session_id: str, request_id: str, user_id: str = Depends(get_limited_user),
                 service=Depends(get_chat_service)):
    """恢复最后一条失败请求，重新校验JWT和会话租约；不会再插入一条用户原文。"""
    try:
        turn = await service.resume_turn(session_id, request_id, user_id)
    except LookupError as error:
        raise HTTPException(404, str(error)) from error
    except (DuplicateRequestError, SessionBusyError, ValueError) as error:
        raise HTTPException(409, str(error)) from error
    except SQLAlchemyError as error:
        raise HTTPException(503, "检查点数据库暂不可用") from error
    return stream_response(service, turn, user_id)
