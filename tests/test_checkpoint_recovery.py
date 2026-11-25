"""真实LangGraph+真实SQL恢复测试；可控模型只负责制造指定步骤失败，不模拟检查点实现。"""
import asyncio
from typing_extensions import TypedDict

import pytest
from langgraph.graph import StateGraph, START, END

from memory.checkpointer import SqlCheckpointSaver


class Work(TypedDict):
    """最小可恢复图状态，工具结果与最终回答分开。"""
    result: str
    answer: str


def test_sql_checkpoint_survives_new_saver_and_skips_completed_node(repository):
    async def run():
        counts = {"lookup": 0, "answer": 0}

        def lookup(state):
            """模拟已经成功的工具读取；恢复时不应再次调用。"""
            counts["lookup"] += 1
            return {"result": "X10"}

        def answer(state):
            """第一次生成故意失败，第二次复用持久化的工具结果完成。"""
            counts["answer"] += 1
            if counts["answer"] == 1:
                raise RuntimeError("生成阶段失败")
            return {"answer": state["result"]}

        def build(saver):
            """重建图与保存器，以真实SQL验证恢复不依赖旧Python对象。"""
            return StateGraph(Work).add_node("lookup", lookup).add_node("answer", answer).add_edge(
                START, "lookup").add_edge("lookup", "answer").add_edge("answer", END).compile(checkpointer=saver)

        config = {"configurable": {"thread_id": "recover"}}
        first = build(SqlCheckpointSaver(repository.engine))
        with pytest.raises(RuntimeError):
            await first.ainvoke({}, config, durability="sync")
        second = build(SqlCheckpointSaver(repository.engine))
        result = await second.ainvoke(None, config, durability="sync")
        assert result["answer"] == "X10"
        assert counts == {"lookup": 1, "answer": 2}
        history = [item async for item in second.aget_state_history(config)]
        assert len(history) >= 4
        assert (await second.aget_state({"configurable": {"thread_id": "other"}})).values == {}
    asyncio.run(run())


def test_pending_writes_keep_success_and_allow_error_replacement(repository):
    from langgraph.checkpoint.base import empty_checkpoint
    saver = SqlCheckpointSaver(repository.engine)
    config = saver.put({"configurable": {"thread_id": "writes"}}, empty_checkpoint(), {"source": "input", "step": 0, "parents": {}}, {})
    saver.put_writes(config, [("value", "first")], "a", "2")
    saver.put_writes(config, [("value", "second")], "a", "2")
    saver.put_writes(config, [("value", "before")], "b", "1")
    saver.put_writes(config, [("__error__", "first error")], "a", "2")
    saver.put_writes(config, [("__error__", "new error")], "a", "2")
    writes = saver.get_tuple(config).pending_writes
    assert writes == [("b", "value", "before"), ("a", "__error__", "new error"), ("a", "value", "first")]
    saver.delete_thread("writes")
    assert saver.get_tuple(config) is None


def test_real_agent_http_resume_does_not_repeat_tool_or_user_message(repository):
    async def run():
        import httpx
        from fastapi import FastAPI
        from langchain_core.messages import AIMessage
        from langchain_core.tools import tool
        from agent.react_agent import ReactAgent
        from api.chat import router as chat_router
        from api.memory import router as memory_router
        from api.deps import get_chat_service, get_history_repository
        from memory.session_lease import SessionLeaseStore
        from services.chat_service import ChatService
        from tests.test_agent_runtime import ScriptedModel
        from utils.security import create_token

        count = {"tool": 0}

        @tool
        def lookup_device() -> str:
            """读取已授权的测试设备型号。"""
            count["tool"] += 1
            return "X10"

        def build(model):
            """每次重建Agent与服务，检查恢复依赖SQL而非原对象的内存。"""
            agent = ReactAgent(checkpointer=SqlCheckpointSaver(repository.engine), model=model,
                               tools=[lookup_device], system_prompt="请查询设备并回答")
            return ChatService(agent, SessionLeaseStore(), repository)

        # 第一次模型只给出工具调用，后续生成因响应耗尽而失败。
        first_model = ScriptedModel(responses=[AIMessage(content="", tool_calls=[{
            "name": "lookup_device", "args": {}, "id": "lookup-1",
        }])])
        holder = {"service": build(first_model)}
        app = FastAPI()
        app.include_router(chat_router)
        app.include_router(memory_router)
        app.dependency_overrides[get_chat_service] = lambda: holder["service"]
        app.dependency_overrides[get_history_repository] = lambda: repository
        owner = {"Authorization": "Bearer " + create_token("checkpoint-owner")}
        other = {"Authorization": "Bearer " + create_token("checkpoint-other")}
        path = "/conversations/recover/requests/request-1"
        async with httpx.AsyncClient(transport=httpx.ASGITransport(app=app), base_url="http://test") as client:
            first = await client.post("/chat", json={"session_id": "recover", "request_id": "request-1", "message": "查询设备"}, headers=owner)
            assert '"type": "error"' in first.text and '"resumable": true' in first.text
            assert count["tool"] == 1
            assert (await client.get(path + "/checkpoints", headers=other)).status_code == 404
            assert (await client.post(path + "/resume", headers=other)).status_code == 404
            holder["service"] = build(ScriptedModel(responses=[AIMessage(content="您的设备是X10")]))
            states = await client.get(path + "/checkpoints", headers=owner)
            assert states.status_code == 200 and len(states.json()["checkpoints"]) >= 3
            resumed = await client.post(path + "/resume", headers=owner)
            assert resumed.status_code == 200 and "您的设备是X10" in resumed.text
            assert count["tool"] == 1
            rows = (await client.get("/conversations/recover/messages", headers=owner)).json()["messages"]
            assert len(rows) == 2 and all(row["status"] == "completed" for row in rows)
            assert (await client.post(path + "/resume", headers=owner)).status_code == 409
    asyncio.run(run())


def test_parallel_pending_success_not_repeated(repository):
    async def run():
        counts = {"a": 0, "b": 0}

        def a(state):
            """同一超步中成功分支的结果应写入pending writes并在恢复时复用。"""
            counts["a"] += 1
            return {"result": "成功结果"}

        def b(state):
            """并行分支首次失败，恢复只能重试此分支。"""
            counts["b"] += 1
            if counts["b"] == 1:
                raise RuntimeError("并行分支失败")
            return {"answer": "恢复成功"}

        builder = StateGraph(Work).add_node("a", a).add_node("b", b).add_edge(START, "a").add_edge(START, "b").add_edge("a", END).add_edge("b", END)
        config = {"configurable": {"thread_id": "parallel"}}
        with pytest.raises(RuntimeError):
            await builder.compile(checkpointer=SqlCheckpointSaver(repository.engine)).ainvoke({}, config, durability="sync")
        result = await builder.compile(checkpointer=SqlCheckpointSaver(repository.engine)).ainvoke(None, config, durability="sync")
        assert counts == {"a": 1, "b": 2}
        assert result == {"result": "成功结果", "answer": "恢复成功"}
    asyncio.run(run())
