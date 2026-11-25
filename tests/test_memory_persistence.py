import asyncio
from types import SimpleNamespace
from unittest.mock import AsyncMock, Mock

import pytest

from db.memory_repository import DuplicateRequestError
from memory.batch_summary import BatchSummarizer
from memory.long_term import extract_memories
from services.chat_types import SessionLeaseError
from services.chat_types import CompressionStatus, ContextBudgetError
from memory.budget import message_tokens
from tests.memory_helpers import service, complete










def test_archive_survives_new_service_and_short_store(repository):
    async def run():
        first = service(repository)
        await complete(first, "same", "我用的是X10")
        second = service(repository)
        turn = await complete(second, "same", "我刚才说的型号是什么？")
        assert "X10" in str(turn.messages)
        rows = repository.history(turn.scope, "1001")
        assert len(rows) == 4
        assert all(row["status"] == "completed" for row in rows)
        repository.initialize()
        assert len(repository.history(turn.scope, "1001")) == 4
    asyncio.run(run())


def test_long_term_recall_across_sessions_and_user_isolation(repository):
    async def run():
        chat = service(repository)
        await complete(chat, "old", "我用的是X10，我家是木地板")
        recalled = await complete(chat, "new", "我的机器人拖地要注意什么？")
        assert "X10" in str(recalled.messages)
        assert "木地板" in str(recalled.messages)
        stranger = await complete(chat, "new", "我的机器人拖地要注意什么？", user="1002")
        assert "X10" not in str(stranger.messages)
        assert repository.history(recalled.scope, "1002") == []
        assert repository.list_memories("1002") == []
    asyncio.run(run())


def test_correction_and_delete_keep_source_and_audit(repository):
    async def run():
        chat = service(repository)
        await complete(chat, "one", "我用的是X10")
        corrected = await complete(chat, "two", "实际型号是B8")
        memories = repository.list_memories("1001")
        assert len(memories) == 1
        assert memories[0]["value"] == "B8"
        assert memories[0]["source_message_id"] == corrected.user_message_id
        assert not repository.delete_memory("1002", memories[0]["id"])
        assert repository.delete_memory("1001", memories[0]["id"])
        assert repository.list_memories("1001") == []
        assert repository.history(corrected.scope, "1001")
    asyncio.run(run())


def test_explicit_forget_updates_projection_not_original_archive(repository):
    async def run():
        chat = service(repository)
        await complete(chat, "one", "我用的是X10")
        turn = await complete(chat, "two", "忘记我的设备型号")
        assert repository.list_memories("1001") == []
        assert "X10" not in str((await complete(chat, "three", "我的设备型号是什么？")).messages)
        assert repository.history(turn.scope, "1001")
    asyncio.run(run())


def test_duplicate_request_does_not_append_again(repository):
    async def run():
        chat = service(repository)
        turn = await complete(chat, "same", "问题", request_id="retry-1")
        with pytest.raises(DuplicateRequestError):
            await chat.open_turn("same", "问题", "1001", request_id="retry-1")
        assert len(repository.history(turn.scope, "1001")) == 2
        assert not chat.leases.owns(turn.scope, turn.owner)
    asyncio.run(run())


def test_explicit_remembered_field_can_be_corrected_without_stale_note(repository):
    async def run():
        chat = service(repository)
        await complete(chat, "first", "记住：我用的是X10")
        await complete(chat, "correction", "实际型号是B8")
        turn = await complete(chat, "recall", "我的设备型号是什么？")
        assert "B8" in str(turn.messages) and "X10" not in str(turn.messages)
        assert len(repository.list_memories("1001")) == 1
    asyncio.run(run())


def test_compression_keeps_recent_question_with_its_answer(repository):
    async def run():
        chat = service(repository)
        await complete(chat, "pairs", "早期内容")
        await complete(chat, "pairs", "最近的问题必须保留")
        turn = await complete(chat, "pairs", "第三个问题")
        assert turn.compression == CompressionStatus.SUCCESS
        assert {"role": "user", "content": "最近的问题必须保留"} in turn.messages
        assert turn.messages[-2] == {"role": "assistant", "content": "测试回答"}
        assert len(repository.history(turn.scope, "1001")) == 6
    asyncio.run(run())


def test_database_lease_protects_even_with_separate_short_stores(repository):
    async def run():
        first = service(repository)
        second = service(repository)
        turn = await first.open_turn("same", "第一问", "1001")
        from services.chat_types import SessionBusyError
        with pytest.raises(SessionBusyError):
            await second.open_turn("same", "第二问", "1001")
        assert len(repository.history(turn.scope, "1001")) == 1
        await first.close_turn(turn)
    asyncio.run(run())


def test_summary_only_advances_watermark_never_deletes_originals(repository):
    async def run():
        chat = service(repository)
        for text in ["消息一", "消息二", "消息三"]:
            turn = await complete(chat, "session", text)
        history = repository.history(turn.scope, "1001")
        assert len(history) == 6
        assert chat._summarizer.ainvoke.await_count >= 1
        assert repository.claim(turn.scope, "1001", "session", "inspect", 120)
        snapshot = repository.snapshot(turn.scope, "inspect")
        assert snapshot["summary"] == "阶段摘要"
        assert snapshot["summary_upto"] > 0
        assert len(snapshot["pending"]) < len(history)
        repository.release(turn.scope, "inspect")
    asyncio.run(run())


def test_second_batch_failure_leaves_old_summary_and_all_originals(repository):
    async def run():
        chat = service(repository, summary_input_budget=900, summary_budget=100)
        for text in ["字" * 350, "另" * 150]:
            await complete(chat, "batch", text)
        chat._summarizer.ainvoke.reset_mock()
        chat._summarizer.ainvoke.side_effect = [
            SimpleNamespace(content="第一批摘要"), RuntimeError("第二批失败"),
        ]
        turn = await complete(chat, "batch", "继续")
        assert turn.compression == CompressionStatus.FAILED
        assert len(repository.history(turn.scope, "1001")) == 6
        assert repository.claim(turn.scope, "1001", "batch", "inspect", 120)
        snapshot = repository.snapshot(turn.scope, "inspect")
        assert snapshot["summary"] == ""
        assert snapshot["summary_upto"] == 0
        repository.release(turn.scope, "inspect")
    asyncio.run(run())


def test_failed_compression_can_continue_with_bounded_context(repository):
    async def run():
        chat = service(repository, context_budget=500, summary_budget=100, summary_input_budget=2000)
        for text in ["旧" * 100, "另" * 100]:
            await complete(chat, "session", text)
        chat._summarizer.ainvoke.side_effect = RuntimeError("摘要不可用")
        turn = await complete(chat, "session", "新问题")
        assert message_tokens(turn.messages) <= 500
        assert turn.messages[-1]["content"] == "新问题"
        assert len(repository.history(turn.scope, "1001")) == 6
    asyncio.run(run())


def test_persistent_summary_cooldown_starts_after_failure(repository):
    async def run():
        chat = service(repository)
        clock = Mock(return_value=100)
        chat._clock = clock
        chat.retry_cooldown = 30
        await complete(chat, "cooldown", "早期问题")
        await complete(chat, "cooldown", "另一个问题")

        async def delayed_failure(prompt):
            """用可控时钟模拟摘要等了40秒后失败，不让测试真实等待。"""
            clock.return_value = 140
            raise TimeoutError("模拟延迟失败")

        chat._summarizer.ainvoke.side_effect = delayed_failure
        turn = await complete(chat, "cooldown", "继续")
        assert turn.compression == CompressionStatus.FAILED
        assert chat._retry_after[turn.scope] == 170
        calls = chat._summarizer.ainvoke.await_count
        clock.return_value = 169
        cooled = await complete(chat, "cooldown", "再问")
        assert cooled.compression == CompressionStatus.COOLDOWN
        assert chat._summarizer.ainvoke.await_count == calls
        clock.return_value = 170
        chat._summarizer.ainvoke.side_effect = None
        retried = await complete(chat, "cooldown", "恢复后再问")
        assert retried.compression == CompressionStatus.SUCCESS
    asyncio.run(run())


def test_old_history_retrieval_is_scoped_and_bounded(repository):
    async def run():
        chat = service(repository)
        await complete(chat, "old", "上次充电问题最终通过更换充电座解决")
        turn = await complete(chat, "new", "上次充电是怎么解决的？")
        assert "更换充电座" in str(turn.messages)
        stranger = await complete(chat, "other", "上次充电是怎么解决的？", user="1002")
        assert "更换充电座" not in str(stranger.messages)
    asyncio.run(run())


def test_failed_agent_keeps_user_record_with_failed_status(repository):
    class BrokenAgent:
        async def execute_stream(self, messages, user_id):
            raise RuntimeError("模型失败")
            yield

    async def run():
        chat = service(repository)
        chat.agent = BrokenAgent()
        turn = await chat.open_turn("failure", "问题", "1001")
        with pytest.raises(RuntimeError):
            async for _ in chat.reply_turn(turn, "1001"):
                pass
        history = repository.history(turn.scope, "1001")
        assert len(history) == 1 and history[0]["status"] == "failed"
    asyncio.run(run())


def test_http_history_and_memory_management_enforce_jwt_user(repository):
    async def run():
        import httpx
        from fastapi import FastAPI
        from api.chat import router as chat_router
        from api.memory import router as memory_router
        from api.deps import get_chat_service, get_history_repository
        from utils.security import create_token

        chat = service(repository)
        app = FastAPI()
        app.include_router(chat_router)
        app.include_router(memory_router)
        app.dependency_overrides[get_chat_service] = lambda: chat
        app.dependency_overrides[get_history_repository] = lambda: repository
        owner = {"Authorization": f"Bearer {create_token('memory-owner')}"}
        stranger = {"Authorization": f"Bearer {create_token('memory-stranger')}"}
        async with httpx.AsyncClient(transport=httpx.ASGITransport(app=app), base_url="http://test") as client:
            body = {"message": "我用的是X10", "session_id": "same", "request_id": "http-1"}
            response = await client.post("/chat", json=body, headers=owner)
            assert response.status_code == 200 and "[DONE]" in response.text
            repeated = await client.post("/chat", json=body, headers=owner)
            assert repeated.status_code == 409
            response = await client.get("/memories", headers=owner)
            memory_id = response.json()["memories"][0]["id"]
            assert (await client.get("/memories", headers=stranger)).json()["memories"] == []
            assert (await client.delete(f"/memories/{memory_id}", headers=stranger)).status_code == 404
            assert (await client.get("/conversations/same/messages", headers=stranger)).json()["messages"] == []
            page = (await client.get("/conversations/same/messages?limit=1", headers=owner)).json()
            assert page["messages"][0]["role"] == "assistant"
            before = page["next_before_id"]
            older = (await client.get(f"/conversations/same/messages?before_id={before}", headers=owner)).json()
            assert older["messages"][0]["content"] == "我用的是X10"
            assert (await client.delete(f"/memories/{memory_id}", headers=owner)).status_code == 200
            assert (await client.get("/memories", headers=owner)).json()["memories"] == []
    asyncio.run(run())


def test_closing_before_generation_marks_request_failed(repository):
    async def run():
        chat = service(repository)
        turn = await chat.open_turn("closed", "问题", "1001")
        await chat.close_turn(turn)
        assert repository.history(turn.scope, "1001")[0]["status"] == "failed"
    asyncio.run(run())


def test_expired_database_owner_cannot_release_new_owner(repository):
    clock = Mock(return_value=100)
    repository.clock = clock
    scope = "1001:expiry"
    assert repository.claim(scope, "1001", "expiry", "old", 10)
    repository.start_request(scope, "1001", "old", "old-request", "原文", [])
    clock.return_value = 111
    assert repository.claim(scope, "1001", "expiry", "new", 10)
    assert not repository.release(scope, "old")
    repository.snapshot(scope, "new")
    assert repository.history(scope, "1001")[0]["status"] == "failed"


def test_long_term_does_not_extract_assistant_generated_claims(repository):
    class FabricatingAgent:
        async def execute_stream(self, messages, user_id):
            yield {"type": "answer", "content": "我用的是BAD1"}

    async def run():
        chat = service(repository)
        chat.agent = FabricatingAgent()
        await complete(chat, "source", "你好")
        assert repository.list_memories("1001") == []
    asyncio.run(run())


def test_case_distinct_users_do_not_share_memory(repository):
    async def run():
        chat = service(repository)
        await complete(chat, "session", "我用的是X10", user="UserA")
        turn = await complete(chat, "session", "我的设备型号是什么", user="usera")
        assert "X10" not in str(turn.messages)
        assert repository.list_memories("usera") == []
    asyncio.run(run())


def test_sql_transaction_rolls_back_partial_assistant_write(repository):
    async def run():
        from sqlalchemy import event
        chat = service(repository)
        turn = await chat.open_turn("rollback", "问题", "1001")

        def break_completion(connection, cursor, statement, parameters, context, executemany):
            """在助手INSERT之后、请求完成UPDATE时注入失败，验证整个事务真实回滚。"""
            values = parameters.values() if isinstance(parameters, dict) else parameters
            if statement.upper().startswith("UPDATE MEMORY_TURNS") and "completed" in values:
                raise RuntimeError("模拟完成标记写入失败")

        event.listen(repository.engine, "before_cursor_execute", break_completion)
        try:
            with pytest.raises(RuntimeError):
                async for _ in chat.reply_turn(turn, "1001"):
                    pass
        finally:
            event.remove(repository.engine, "before_cursor_execute", break_completion)
        rows = repository.history(turn.scope, "1001")
        assert len(rows) == 1 and rows[0]["role"] == "user" and rows[0]["status"] == "failed"
    asyncio.run(run())


def test_structured_facts_not_crowded_out_by_recent_notes(repository):
    async def run():
        chat = service(repository)
        await complete(chat, "facts", "我用的是X10")
        for index in range(5):
            await complete(chat, f"notes{index}", f"记住：型号相关笔记{index}")
        rows = repository.search_memories("1001", ["型号"], ["device_model"], limit=3)
        assert any(row["memory_key"] == "device_model" and row["value"] == "X10" for row in rows)
        assert repository.search_memories("1001", ["%' OR 1=1 --"]) == []
    asyncio.run(run())


def test_stale_owner_cannot_commit_or_write(repository):
    scope = "1001:test"
    assert repository.claim(scope, "1001", "test", "owner", 120)
    turn_id, _ = repository.start_request(scope, "1001", "owner", "request", "原文", [])
    snapshot = repository.snapshot(scope, "owner")
    repository.release(scope, "owner")
    assert repository.claim(scope, "1001", "test", "other", 120)
    with pytest.raises(SessionLeaseError):
        repository.finish_request(scope, "owner", turn_id, "过期回答")
    with pytest.raises(SessionLeaseError):
        repository.commit_summary(scope, "owner", snapshot["revision"], snapshot["pending"][0]["id"], "过期摘要")
    assert len(repository.history(scope, "1001")) == 1


def test_single_long_message_is_split_without_dropping_text():
    async def run():
        model = SimpleNamespace(ainvoke=AsyncMock(return_value=SimpleNamespace(content="摘要")))
        summarizer = BatchSummarizer(model, input_budget=900, output_budget=100, max_batches=20)
        original = "甲乙丙丁" * 150
        summary, count = await summarizer.summarize("", [{"id": 1, "role": "user", "content": original}])
        assert summary == "摘要" and count > 1
        fragments = [call.args[0].split("本批历史片段：\n", 1)[1] for call in model.ainvoke.call_args_list]
        assert "".join(fragments) == "[消息1 user] " + original
        assert all(len(call.args[0].encode("utf-8")) + 12 <= 900 for call in model.ainvoke.call_args_list)
    asyncio.run(run())


@pytest.mark.parametrize("text", ["如果我用的是X10该怎么办", "别人用的是X10", "不要记住，我用的是X10"])
def test_hypothetical_or_optout_not_persisted(text):
    assert extract_memories(text) == []


@pytest.mark.parametrize("text", ["我用的是X10吗？", "我家是木地板吗？", "请举例：我的型号是X10"])
def test_questions_and_examples_are_not_user_facts(text):
    assert extract_memories(text) == []


def test_batch_count_limit_stops_without_partial_result():
    async def run():
        model = SimpleNamespace(ainvoke=AsyncMock(return_value=SimpleNamespace(content="摘要")))
        summarizer = BatchSummarizer(model, 900, 100, max_batches=1)
        with pytest.raises(ContextBudgetError):
            await summarizer.summarize("", [{"id": 1, "role": "user", "content": "长" * 1000}])
        assert model.ainvoke.await_count == 1
    asyncio.run(run())
