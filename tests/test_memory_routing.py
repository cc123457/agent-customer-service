"""真实SQL验证统一读写路由和所有模型入口的遗忘约束；模型替身只控制不可控输出。"""
import asyncio
import json
from types import SimpleNamespace
from unittest.mock import AsyncMock, Mock

import pytest

from memory.request_analyzer import MemoryRequestAnalyzer
from memory.history_retrieval import HistoryRetriever
from tests.memory_helpers import complete, service


def test_single_analysis_combines_read_write_and_does_not_promote_question():
    async def run():
        analyzer = MemoryRequestAnalyzer()
        plan = await analyzer.analyze("给我讲排查步骤时，我要求的回答详细程度是什么？")
        assert plan.operations == []
        text = "我怕机器工作时吵醒午睡的人，照以前那样挑一个。"
        payload = {"facts": [{"key": "note", "value": "怕机器工作时吵醒午睡的人", "entity": "profile",
                    "evidence": text, "valid_at": "", "action": "upsert"}],
                   "read": {"search_notes": True, "search_history": True, "query": "午睡 噪声 选购"}}
        model = Mock()
        model.bind.return_value = model
        model.ainvoke = AsyncMock(return_value=SimpleNamespace(content=json.dumps(payload)))
        # 非明确规则表达，读取与写入共享一个请求。
        plan = await MemoryRequestAnalyzer(model).analyze(text)
        assert len(plan.operations) == 1 and plan.read.search_history
        assert model.ainvoke.await_count == 1
        assert (await analyzer.analyze("我用的是X10，怎么充电？")).operations[0].value == "X10"
    asyncio.run(run())


@pytest.mark.parametrize("forget", ["请忘记我之前的型号", "请删除我之前的设备型号记录", "忘记我的型号"])
def test_forget_blocks_recent_summary_dated_history_and_allows_new_source(repository, forget):
    async def run():
        chat = service(repository)
        original = await complete(chat, "a", "我用的是OLD-719。")
        assert repository.claim(original.scope, "1001", "a", "summary", 3600)
        snapshot = repository.snapshot(original.scope, "summary")
        assert repository.commit_summary(original.scope, "summary", snapshot["revision"],
                                         snapshot["recent"][-1]["id"], "用户设备是OLD-719")
        repository.release(original.scope, "summary")
        deletion = await complete(chat, "b", forget)
        assert deletion.direct_answer and repository.active_memories("1001") == []
        current = await chat.open_turn("a", "我之前使用的型号是什么？", "1001")
        assert "OLD-719" not in str(current.messages)
        assert "OLD-719" not in str(HistoryRetriever(repository).search("1001", "OLD-719", after=0))
        await chat.close_turn(current)
        await complete(chat, "c", "我用的是NEW-820。")
        current = await chat.open_turn("c", "我的型号是什么？", "1001")
        assert "NEW-820" in str(current.messages) and "OLD-719" not in str(current.messages)
        await chat.close_turn(current)
        # 原文仍能由本人档案接口查看，语义遗忘没有伪装成物理删除。
        assert "OLD-719" in str(repository.history(original.scope, "1001"))
    asyncio.run(run())


def test_scoped_deletion_keeps_other_device_and_other_user(repository):
    async def run():
        chat = service(repository)
        await complete(chat, "a", "客厅设备型号是LIVE-39。")
        await complete(chat, "a", "卧室设备型号是GONE-48。")
        await complete(chat, "a", "我用的是OTHER-57。", user="other")
        await complete(chat, "a", "忘记卧室设备型号，帮我找出客厅上次的维修方案。")
        assert [row["value"] for row in repository.active_memories("1001")] == ["LIVE-39"]
        assert repository.active_memories("other")[0]["value"] == "OTHER-57"
        assert not repository.delete_memory("other", repository.active_memories("1001")[0]["id"])
    asyncio.run(run())


def test_note_delete_ambiguity_and_future_budget_cancellation(repository):
    async def run():
        from memory.fact_rules import effective_timestamp
        chat = service(repository)
        repository.clock = lambda: effective_timestamp("2026-11-25")
        await complete(chat, "a", "记住：配件预算上限为500元。")
        await complete(chat, "a", "记住：设备预算上限为5000元。")
        turn = await complete(chat, "a", "忘记我的预算。")
        assert "请说明" in turn.direct_answer and len(repository.active_memories("1001")) == 2
        await complete(chat, "a", "忘记我的配件预算。")
        assert len(repository.active_memories("1001")) == 1
        await complete(chat, "a", "从2026-12-10起，记住：配件预算上限为600元。")
        await complete(chat, "a", "忘记配件预算。")
        repository.clock = lambda: effective_timestamp("2026-12-11")
        assert not any("600" in row["value"] for row in repository.active_memories("1001"))
    asyncio.run(run())


def test_routes_skip_unneeded_embedding_and_find_imperative_history(repository):
    async def run():
        semantic = Mock()
        chat = service(repository, semantic_index=semantic)
        await complete(chat, "old", "电机检修接收单是RECEIPT-71，最终领件码PICK-82。")
        turn = await chat.open_turn("new", "把电机检修的接收单和最终领件码一并告诉我。", "1001")
        assert turn.memory_analysis["search_history"] and "PICK-82" in str(turn.messages)
        await chat.close_turn(turn)
        semantic.reset_mock()
        turn = await complete(chat, "new", "你好")
        assert not turn.memory_analysis["search_history"]
        semantic.sync_user.assert_not_called()
        semantic.search.assert_not_called()
    asyncio.run(run())


def test_model_failure_uses_bounded_fallback_and_only_one_attempt():
    async def run():
        model = Mock()
        model.bind.return_value = model
        model.ainvoke = AsyncMock(return_value=SimpleNamespace(content="broken JSON"))
        plan = await MemoryRequestAnalyzer(model).analyze("我对夜间的机器响声很敏感")
        assert plan.degraded and plan.read.search_history and plan.operations == []
        assert model.ainvoke.await_count == 1
    asyncio.run(run())


def test_forget_epoch_rejects_inflight_output_and_old_resume(repository):
    async def run():
        chat = service(repository)
        await complete(chat, "old", "我用的是SECRET-72。")
        pending = await chat.open_turn("old", "型号是什么", "1001", "stale")
        await complete(chat, "delete", "忘记我的型号")
        with pytest.raises(ValueError, match="上下文已失效"):
            async for _ in chat.reply_turn(pending, "1001"):
                pytest.fail("过期上下文不得发出回答")
        assert repository.claim(pending.scope, "1001", "old", "resume", 3600)
        try:
            with pytest.raises(ValueError, match="上下文已失效"):
                repository.resume_request(pending.scope, "1001", "resume", "stale")
        finally:
            repository.release(pending.scope, "resume")
    asyncio.run(run())


def test_api_delete_uses_same_policy_and_cancels_entire_transaction_on_error(repository, monkeypatch):
    async def run():
        chat = service(repository)
        old = await complete(chat, "a", "我用的是SAFE-29。")
        row = repository.active_memories("1001")[0]
        import db.memory_privacy as privacy
        original = privacy.forget
        def failed(*args):
            """模拟屏障写入失败，事实及本次原文必须和事务一起回滚。"""
            raise RuntimeError("SQL unavailable")
        monkeypatch.setattr(privacy, "forget", failed)
        with pytest.raises(RuntimeError):
            await chat.open_turn("delete", "忘记我的型号", "1001")
        assert repository.active_memories("1001")[0]["value"] == "SAFE-29"
        assert repository.history("1001:delete", "1001") == []
        monkeypatch.setattr(privacy, "forget", original)
        assert repository.delete_memory("1001", row["id"])
        assert repository.access_policy("1001").version == 1
        assert HistoryRetriever(repository).search("1001", "SAFE-29") == []
    asyncio.run(run())


def test_embedding_space_distinguishes_same_dimension_models_and_wrappers():
    from model.embedding_space import embedding_space
    a = SimpleNamespace(model="text-embedding-v3", dimensions=1024, openai_api_base="same")
    b = SimpleNamespace(model="text-embedding-v4", dimensions=1024, openai_api_base="same")
    assert embedding_space(a) != embedding_space(b)
    assert embedding_space(SimpleNamespace(delegate=a)) == embedding_space(a)


def test_api_id_delete_does_not_delete_another_entity(repository):
    async def run():
        chat = service(repository)
        await complete(chat, "a", "我用的是PROFILE-54。")
        await complete(chat, "a", "卧室设备型号是BED-63。")
        profile = next(row for row in repository.active_memories("1001") if row["entity_id"] == "profile")
        assert repository.delete_memory("1001", profile["id"])
        assert [row["value"] for row in repository.active_memories("1001")] == ["BED-63"]
    asyncio.run(run())


def test_legacy_soft_delete_migration_is_idempotent(repository):
    async def run():
        from sqlalchemy import update
        from db.memory_schema import memories
        chat = service(repository)
        await complete(chat, "a", "我用的是LEGACY-81。")
        row = repository.active_memories("1001")[0]
        with repository.engine.begin() as connection:
            connection.execute(update(memories).where(memories.c.id == row["id"]).values(status="deleted"))
        repository.initialize()
        version = repository.access_policy("1001").version
        repository.initialize()
        assert version == 1 and repository.access_policy("1001").version == version
        assert HistoryRetriever(repository).search("1001", "LEGACY-81") == []
    asyncio.run(run())


def test_forget_during_analysis_rejects_stale_preparation(repository):
    async def run():
        chat = service(repository)
        await complete(chat, "a", "我用的是RACE-24。")
        row = repository.active_memories("1001")[0]
        class RacingAnalyzer(MemoryRequestAnalyzer):
            async def analyze(self, *args):
                """模拟别的会话在模型分析等待期间完成同用户删除。"""
                result = await super().analyze(*args)
                repository.delete_memory("1001", row["id"])
                return result
        chat.request_analyzer = RacingAnalyzer()
        with pytest.raises(ValueError, match="分析期间"):
            await chat.open_turn("b", "我之前的型号？", "1001")
        assert repository.history("1001:b", "1001") == []
    asyncio.run(run())


def test_redis_warm_fact_cannot_resurrect_after_forgetting(repository, redis_client):
    async def run():
        from memory.fact_cache import RedisFactCache
        from uuid import uuid4
        prefix = "memory:routing-test:" + uuid4().hex
        repository.fact_cache = RedisFactCache(redis_client, key_prefix=prefix)
        try:
            chat = service(repository)
            await complete(chat, "a", "我用的是CACHE-46。")
            for _ in range(2):
                assert repository.search_memories("1001", keys=["device_model"])[0]["value"] == "CACHE-46"
            await complete(chat, "delete", "请删除我之前的设备型号记录。")
            assert repository.search_memories("1001", keys=["device_model"]) == []
        finally:
            for key in redis_client.scan_iter(match=prefix + ":*"):
                redis_client.delete(key)
    asyncio.run(run())


def test_question_is_rejected_even_if_model_proposes_grounded_note():
    async def run():
        text = "我对夜间噪声很敏感吗？"
        payload = {"facts": [{"key": "note", "value": "我对夜间噪声很敏感", "entity": "profile",
                    "evidence": text, "valid_at": "", "action": "upsert"}]}
        model = Mock()
        model.bind.return_value = model
        model.ainvoke = AsyncMock(return_value=SimpleNamespace(content=json.dumps(payload)))
        plan = await MemoryRequestAnalyzer(model).analyze(text)
        assert plan.operations == [] and plan.degraded
    asyncio.run(run())


def test_analysis_input_budget_does_not_truncate_and_write_partial_fact():
    async def run():
        model = Mock()
        plan = await MemoryRequestAnalyzer(model, input_budget=1).analyze("我对机器运行声音特别敏感。")
        assert plan.degraded and not plan.model_called and plan.operations == []
        model.bind.assert_not_called()
    asyncio.run(run())


def test_same_model_two_devices_do_not_share_forgetting_scope(repository):
    async def run():
        chat = service(repository)
        await complete(chat, "a", "客厅设备型号是SAME-75。")
        await complete(chat, "b", "卧室设备型号是SAME-75。")
        await complete(chat, "a", "忘记客厅的设备型号。")
        found = HistoryRetriever(repository).search("1001", "卧室 SAME-75")
        assert found and all("客厅" not in str(turn["messages"]) for turn in found)
        assert repository.active_memories("1001")[0]["entity_id"] == "卧室"
    asyncio.run(run())


def test_mixed_preference_read_write_accepts_literal_sentence_punctuation():
    async def run():
        text = "我更在意机器安静一些，帮我找上次客厅检修的最终方案。"
        model = Mock()
        model.bind.return_value = model
        model.ainvoke = AsyncMock(return_value=SimpleNamespace(content=json.dumps({"facts": [
            {"key": "note", "value": "我更在意机器安静一些", "entity": "profile", "evidence": text, "valid_at": "", "action": "upsert"}]})))
        plan = await MemoryRequestAnalyzer(model).analyze(text)
        assert plan.read.search_history and len(plan.operations) == 1
        assert model.ainvoke.await_count == 1
        text = "我下午需要午睡，不喜欢噪声。"
        model.ainvoke = AsyncMock(return_value=SimpleNamespace(content=json.dumps({"facts": [
            {"key": "note", "value": text, "entity": "profile", "evidence": text, "valid_at": "", "action": "upsert"}]})))
        plan = await MemoryRequestAnalyzer(model).analyze(text)
        assert plan.operations[0].value == text
    asyncio.run(run())


@pytest.mark.parametrize("text,key,value", [
    ("我目前使用的设备标识为UNIT-19，请后续按这个型号答复。", "device_model", "UNIT-19"),
    ("我购买配件的预算最多一千五百元。", "note", "我购买配件的预算最多一千五百元。"),
])
def test_self_statement_does_not_get_skipped_by_known_read_route(text, key, value):
    async def run():
        model = Mock()
        model.bind.return_value = model
        model.ainvoke = AsyncMock(return_value=SimpleNamespace(content=json.dumps({"facts": [
            {"key": key, "value": value, "entity": "profile", "evidence": text, "valid_at": "", "action": "upsert"}]})))
        plan = await MemoryRequestAnalyzer(model).analyze(text)
        assert plan.operations[0].value == value and model.ainvoke.await_count == 1
    asyncio.run(run())


def test_invalid_read_falls_back_without_discarding_independently_valid_fact():
    async def run():
        text = "我对机器运行时的噪声特别敏感。"
        model = Mock()
        model.bind.return_value = model
        payload = {"facts": [{"key": "note", "value": text, "entity": "profile", "evidence": text, "valid_at": "", "action": "upsert"}],
                   "read": {"fact_keys": ["invalid_attribute"]}}
        model.ainvoke = AsyncMock(return_value=SimpleNamespace(content=json.dumps(payload)))
        plan = await MemoryRequestAnalyzer(model).analyze(text)
        assert plan.operations[0].value == text and plan.degraded and plan.read.search_history
        payload["user_id"] = "untrusted"
        plan = await MemoryRequestAnalyzer(model).analyze(text)
        # 重新设置模型返回值后，顶层未知权限/身份字段不能使用部分校验旁路。
        model.ainvoke.return_value = SimpleNamespace(content=json.dumps(payload))
        assert (await MemoryRequestAnalyzer(model).analyze(text)).operations == []
    asyncio.run(run())
