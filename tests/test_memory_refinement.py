"""记忆更正、生效日期、摘要关键原文和参考费用的实际路径回归。"""
import asyncio
import json
from datetime import datetime
from types import SimpleNamespace
from unittest.mock import Mock

import pytest
from sqlalchemy import insert, select, update

from memory.fact_rules import effective_timestamp
from memory.long_term import extract_memories


def test_pending_plan_context_uses_request_clock_and_keeps_original():
    """跨年未到期计划由真实日期规则标记，原文与来源保留；到期及模糊/非法日期不伪标。"""
    from memory.context_builder import format_history_turn, pending_plan_label
    as_of = effective_timestamp("2026-12-31") + 86340
    original = "自2027-01-01开始，我家是木地板。"
    turn = {"created_at": effective_timestamp("2026-11-22"), "session_id": "old", "turn_id": "plan",
            "messages": [{"id": 1, "role": "user", "content": original},
                         {"id": 2, "role": "assistant", "content": "已记录计划"}]}
    text = format_history_turn(turn, 1200, as_of)
    assert "计划尚未生效" in text and "2026-12-31T23:59:00+08:00" in text
    assert original in text and "来源会话:old" in text
    assert not pending_plan_label(original, effective_timestamp("2027-01-01"))
    assert not pending_plan_label("明天改木地板", as_of)
    assert not pending_plan_label("从2026-02-30起改木地板", as_of)
    assert not pending_plan_label("从2027-01-01起改地毯，从2027-02-01起改瓷砖", as_of)


def test_recent_plan_annotation_preserves_budget_and_archive_deduplication():
    """近期整轮使用同样计划标签，仍按完整包装算预算，命中同轮历史不重复注入。"""
    from memory.context_builder import compose_context
    from memory.budget import message_tokens
    original = "从2027-01-01起，我家是木地板。"
    pair = [{"id": 1, "turn_id": "plan", "role": "user", "content": original},
            {"id": 2, "turn_id": "plan", "role": "assistant", "content": "记录明确日期"}]
    recent = pair + [{"id": 3, "turn_id": "current", "role": "user", "content": "当前地面？"}]
    archive = [{"session_id": "old", "turn_id": "plan", "created_at": effective_timestamp("2026-11-22"), "messages": pair}]
    context = compose_context(recent, "", [], archive, 900, as_of=effective_timestamp("2026-12-31"))
    assert str(context).count(original) == 1 and "计划尚未生效" in str(context)
    assert context[-1]["content"] == "当前地面？" and message_tokens(context) <= 900


def save(repository, text, user="1001", session="test"):
    """通过原有用户锁/事务写一轮真实问答，不直接绕过业务持久化。"""
    scope = user + ":" + session
    assert repository.claim(scope, user, session, "writer", 3600)
    turn_id, _ = repository.start_request(scope, user, "writer", str(repository.clock()) + text, text, extract_memories(text))
    repository.finish_request(scope, "writer", turn_id, "已登记")
    repository.release(scope, "writer")


def test_budget_update_uses_one_attribute_and_preserves_events(repository):
    """同对象预算更正只保留一个当前值，其他预算对象/自由笔记不被误删，旧原文仍在。"""
    save(repository, "记住：配件预算上限是2504元。")
    save(repository, "记住：配件预算上限改为2604元。")
    save(repository, "记住：设备预算上限是5000元。")
    save(repository, "记住：对噪声很敏感。")
    rows = repository.active_memories("1001")
    assert len(rows) == 3
    assert sum("2604" in row["value"] for row in rows) == 1
    assert not any("2504" in row["value"] for row in rows)
    assert any("2504" in value for value in repository.superseded_values("1001"))
    assert len(repository.history("1001:test", "1001")) == 8


def test_future_fact_activates_only_at_business_date(repository):
    """未来更正不提前覆盖现值，到UTC+8生效点才换值，版本与向量SQL复核一起更新。"""
    now = effective_timestamp("2026-11-24")
    repository.clock = Mock(return_value=now)
    save(repository, "我家是瓷砖。")
    save(repository, "从2026-12-10起，我家是木地板。")
    assert repository.search_memories("1001", keys=["floor_type"])[0]["value"] == "瓷砖"
    version = repository.memory_version("1001")
    repository.clock.return_value = effective_timestamp("2026-12-10") - .01
    assert repository.active_memories("1001")[0]["value"] == "瓷砖"
    repository.clock.return_value += .01
    rows = repository.active_memories("1001")
    assert rows[0]["value"] == "木地板" and rows[0]["valid_at"] == "2026-12-10"
    assert repository.memory_version("1001") > version
    assert "瓷砖" in repository.superseded_values("1001")
    assert repository.active_memories("other") == []


@pytest.mark.parametrize("operation", ["forget", "later_correction", "api_delete"])
def test_deleted_or_newer_memory_is_not_resurrected_by_old_plan(repository, operation):
    """未来计划不能在用户遗忘、管理删除或更新陈述后复活旧意图。"""
    repository.clock = Mock(return_value=effective_timestamp("2026-11-24"))
    save(repository, "我家是瓷砖。")
    save(repository, "从2026-12-10起，我家是木地板。")
    if operation == "api_delete":
        assert repository.delete_memory("1001", repository.active_memories("1001")[0]["id"])
    else:
        save(repository, "忘记我的地面" if operation == "forget" else "我家是地毯。")
    repository.clock.return_value = effective_timestamp("2026-12-11")
    rows = repository.active_memories("1001")
    assert [row["value"] for row in rows] == (["地毯"] if operation == "later_correction" else [])


def test_forget_future_only_fact_and_invalid_dates(repository):
    """只有未来计划尚无当前行时，也能取消；非法日期/多日期不写成即时事实。"""
    repository.clock = Mock(return_value=effective_timestamp("2026-11-24"))
    save(repository, "从2026-12-10起，我家是木地板。")
    save(repository, "忘记我的地面")
    repository.clock.return_value = effective_timestamp("2026-12-11")
    assert repository.active_memories("1001") == []
    assert extract_memories("从2026-02-30起，我家是木地板。") == []
    assert extract_memories("从2026-12-01起，我家是木地板。从2026-12-02起，我家是瓷砖。") == []
    assert extract_memories("从2026-12-10起，我家是木地板。")[0].valid_at == "2026-12-10"


def test_concurrent_due_reads_apply_one_event(repository):
    """多个读取同时跨生效点，只激活一次并保留单一当前值；这是互斥断言而非容量压测。"""
    from concurrent.futures import ThreadPoolExecutor
    from db.memory_schema import memory_events
    repository.clock = Mock(return_value=effective_timestamp("2026-11-24"))
    save(repository, "我家是瓷砖。")
    save(repository, "从2026-12-10起，我家是木地板。")
    repository.clock.return_value = effective_timestamp("2026-12-10")
    with ThreadPoolExecutor(max_workers=4) as pool:
        results = list(pool.map(lambda _: repository.active_memories("1001"), range(4)))
    assert all([row["value"] for row in rows] == ["木地板"] for rows in results)
    with repository.engine.connect() as connection:
        events = connection.execute(select(memory_events.c.value)).scalars().all()
    assert events.count("木地板") == 1


def test_warm_redis_cache_changes_at_effective_time(repository, redis_client):
    """真实Redis热缓存跨生效点不能继续返回旧值，TTL尚未到期也会换SQL版本。"""
    from uuid import uuid4
    from memory.fact_cache import RedisFactCache
    prefix = "memory:pytest_schedule:" + uuid4().hex
    repository.fact_cache = RedisFactCache(redis_client, key_prefix=prefix)
    repository.clock = Mock(return_value=effective_timestamp("2026-11-24"))
    try:
        save(repository, "我家是瓷砖。")
        save(repository, "从2026-12-10起，我家是木地板。")
        for _ in range(2):
            assert repository.search_memories("1001", keys=["floor_type"])[0]["value"] == "瓷砖"
        assert repository.fact_cache.stats("1001")["hits"] == 1
        repository.clock.return_value = effective_timestamp("2026-12-10")
        assert repository.search_memories("1001", keys=["floor_type"])[0]["value"] == "木地板"
    finally:
        for key in redis_client.scan_iter(match=prefix + ":*"):
            redis_client.delete(key)


def test_summary_restores_missing_ids_without_confusing_prefixes():
    """摘要漏编号时补原文，不把X100当成X10；预算不足显式失败而不是丢掉关键值。"""
    from memory.batch_summary import preserve_identifiers
    from services.chat_types import ContextBudgetError
    source = "用户旧设备为X10。实际型号改为X100。维修编号TK-4422。"
    text = preserve_identifiers("当前型号X100。", source, 1024)
    assert "TK-4422" in text and "旧设备为X10" in text and "实际型号改为X100" in text
    with pytest.raises(ContextBudgetError):
        preserve_identifiers("正常", source, 30)


def test_legacy_hash_budget_rows_are_filtered_before_vector_id_recheck(repository):
    """旧库保留两个不同哈希的预算时，读接口和向量ID复核也不能恢复旧的当前值。"""
    from db.memory_schema import memories
    from db.memory_extensions import fact_details
    save(repository, "记住：配件预算上限是2504元。")
    first = repository.active_memories("1001")[0]
    with repository.engine.begin() as connection:
        connection.execute(update(memories).where(memories.c.id == first["id"]).values(memory_key="note:legacy_old"))
        connection.execute(update(fact_details).where(fact_details.c.memory_id == first["id"]).values(attribute="note:legacy_old"))
    save(repository, "记住：配件预算上限改为2604元。")
    with repository.engine.begin() as connection:
        # 模拟尚未执行新写入整理的旧数据，两个哈希行都还是active。
        connection.execute(update(memories).where(memories.c.id == first["id"]).values(status="active"))
    assert len(repository.list_memories("1001")) == 1
    assert repository.active_memories("1001", [first["id"]]) == []
    assert len(repository.search_memories("1001", terms=["预算"])) == 1


def test_new_acceptance_cases_are_fixed_and_have_real_stored_future_plans():
    """新增日期题的计划在提问前已说出，不能靠剔除未来消息伪装为时间生效验证。"""
    from pathlib import Path
    from evals.build_refinement_cases import build
    from evals.history_metrics import validate_dataset
    saved = json.loads(Path("evals/refinement_cases.json").read_text(encoding="utf-8"))
    assert saved == build()
    assert len(validate_dataset(saved)) == 24
    for case in saved["cases"]:
        if case["category"] == "effective_date":
            plan = next(row for row in case["history"] if row["id"].endswith(":planned"))
            assert plan["created_at"] < case["question_time"]


def test_reference_cost_excludes_diagnostic_and_other_answer_calls():
    """当前方案只承担自身冷启动成本，诊断向量和完整历史回答不能混算进它。"""
    from model.usage import UsageLedger, combine_reports, record_usage, usage_scope
    ledger = UsageLedger()
    with usage_scope(ledger):
        for stage, tokens in (("memory", 100), ("summary", 50), ("full_history", 1000)):
            record_usage("qwen3.5-omni-flash", "chat", {"input_tokens": tokens, "output_tokens": 0}, stage=stage)
        for stage in ("memory_context/documents", "memory_context/query", "diagnostic_retrieval_hybrid/query"):
            record_usage("text-embedding-v4", "embedding", {"total_tokens": 10}, stage=stage)
    result = combine_reports([ledger.report()])
    assert result["memory_cold_reference_cost"] == pytest.approx((150 * 2.2 + 20 * .5) / 1_000_000)
    assert result["full_history_reference_cost"] == pytest.approx(1000 * 2.2 / 1_000_000)
    assert result["actual_paid"] is None


def test_embedding_usage_records_actual_batches_and_preserves_vectors():
    """真实LangChain分批路径在丢弃usage之前计量，不改向量值，不依赖正文猜Token。"""
    from model.factory import embed_model
    from model.usage import MeteredEmbeddingClient, UsageLedger, usage_scope
    calls = []

    def create(input, **kwargs):
        """末端HTTP替身仅控制提供方usage，按每文本2 Token返回。"""
        calls.append(len(input))
        return {"data": [{"embedding": [float(value), 1.0]} for value in input], "usage": {"total_tokens": len(input) * 2}}

    model = embed_model.model_copy(update={"client": MeteredEmbeddingClient(SimpleNamespace(create=create), "text-embedding-v4")})
    ledger = UsageLedger()
    with usage_scope(ledger, "history_index_build"):
        vectors = model.embed_documents([str(number) for number in range(21)])
    assert calls == [10, 10, 1] and [row[0] for row in vectors] == list(range(21))
    report = ledger.report()
    assert len(report["events"]) == 3
    assert report["stages"]["history_index_build"]["input_tokens_reported"] == 42
    assert report["reference_cost_known_subtotal"] == pytest.approx(42 * .5 / 1_000_000)
    assert report["actual_paid"] is None


def test_usage_scopes_propagate_to_thread_and_do_not_leak_between_tasks():
    """并行案例各有自己的费用账本，阶段归属保留，缺失用量不被当作已计价。"""
    from model.usage import UsageLedger, record_usage, usage_scope

    async def one(number):
        """每个异步任务为线程内Embedding工作传递独立账本。"""
        ledger = UsageLedger()
        with usage_scope(ledger, "query"):
            await asyncio.to_thread(record_usage, "text-embedding-v4", "embedding", {"total_tokens": number})
            record_usage("text-embedding-v4", "embedding", {})
        return ledger.report()

    async def scenario():
        """并行运行两个不同用量的案例，避免只测顺序路径。"""
        return await asyncio.gather(one(3), one(9))

    first, second = asyncio.run(scenario())
    assert first["stages"]["query"]["input_tokens_reported"] == 3
    assert second["stages"]["query"]["input_tokens_reported"] == 9
    assert not first["reference_cost_complete"] and first["unpriced_or_missing_calls"] == 1


def test_embedding_total_tokens_is_used_when_optional_prompt_field_is_none():
    """百炼示例只含total_tokens，SDK补出的prompt_tokens=None不能遮掉真实用量。"""
    from model.usage import UsageLedger, record_usage, usage_scope
    ledger = UsageLedger()
    with usage_scope(ledger, "query"):
        record_usage("text-embedding-v4", "embedding", {"prompt_tokens": None, "total_tokens": 7})
    assert ledger.report()["stages"]["query"]["input_tokens_reported"] == 7


def test_access_denied_stops_following_model_calls():
    """一次401/403后同一评测器不再发送下一次请求，错误不会伪装成合法模型输出。"""
    from scripts.evaluate_history import CountedModel, ProviderAccessBlocked, RequestGate

    class Denied(Exception):
        status_code = 403

    class Model:
        model_name = "qwen3.5-omni-flash"

        def __init__(self):
            """只计真实进入替身接口的次数。"""
            self.calls = 0

        async def ainvoke(self, messages):
            """模拟免费额度耗尽的提供方访问拒绝。"""
            self.calls += 1
            raise Denied("模型访问拒绝")

    async def run():
        """首个失败保留原异常，后续只抛本地阻断，不再走SDK。"""
        delegate, events, gate = Model(), [], RequestGate(0)
        model = CountedModel(delegate, events, "summary", gate)
        with pytest.raises(Denied):
            await model.ainvoke([])
        with pytest.raises(ProviderAccessBlocked):
            await model.ainvoke([])
        assert delegate.calls == len(events) == 1 and events[0]["status_code"] == 403

    asyncio.run(run())


def test_json_fence_normalization_keeps_strict_schema_and_no_prose_extraction():
    """只清理外层围栏，多对象、正文解释、字符串布尔和额外字段仍必须被拒绝。"""
    from services.json_payload import json_payload
    from scripts.evaluate_history import ShortAnswer
    good = '{"answer":"X20","abstain":false}'
    for text in (good, "```json\n" + good + "\n```", "\n" + good + "\n```"):
        assert ShortAnswer.model_validate_json(json_payload(text)).answer == "X20"
    for text in ("这是JSON：" + good, good + good, '{"answer":"X20","abstain":"false"}',
                 '{"answer":"X20","abstain":false,"extra":1}'):
        with pytest.raises(ValueError):
            ShortAnswer.model_validate_json(json_payload(text))
