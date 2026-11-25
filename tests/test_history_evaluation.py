"""验证评测标签、隔离与分母，不用写死答案冒充模型质量评估。"""
import json
from pathlib import Path

import pytest

from evals.build_history_cases import build
from evals.history_metrics import answer_score, retrieval_score, validate_dataset, visible_history
from scripts.evaluate_history import (ShortAnswer, aggregate, full_context, planned_filters, retrieval_run, seed_case)


def test_dataset_labels_and_freeze():
    """生成器可重复、标注有效，校准/验收用户不共享，受限数据不属于正确证据。"""
    dataset = build()
    cases = validate_dataset(dataset)
    saved = json.loads((Path(__file__).resolve().parents[1] / "evals/history_cases.json").read_text(encoding="utf-8"))
    assert saved == dataset
    assert len(cases) == 60
    assert sum(case["split"] == "calibration" for case in cases) == 12
    assert sum(case["split"] == "test" for case in cases) == 48
    for case in cases:
        assert case["id"] not in case["question"]
        assert all(alias not in case["question"] for group in case["answer_groups"] for alias in group if "-" in alias)


def test_multi_source_and_negative_metrics():
    """两份证据只找到一份是Hit=1、Recall=.5，负例不应混入正例召回率。"""
    case = next(case for case in build()["cases"] if case["category"] == "multi_session")
    score = retrieval_score(case, ["wrong", case["gold_turn_ids"][0]])
    assert score["hit"] and score["recall"] == .5 and score["reciprocal_rank"] == .5
    assert not score["all_evidence"] and score["invalid_sources"] == 1
    negative = next(case for case in build()["cases"] if case["category"] == "no_answer")
    assert retrieval_score(negative, [negative["history"][0]["id"]])["recall"] is None


def test_answer_boundaries_and_abstention():
    """区分数字/型号前缀、完整字段和拒答，结构合法并不意味着内容正确。"""
    case = {"answer_groups": [["X20"], ["10:45", "十点四十五分"]], "forbidden_answers": [], "should_abstain": False}
    assert answer_score(case, "X20，10：45", False)["passed"]
    assert not answer_score(case, "X200，10:45", False)["passed"]
    assert not answer_score(case, "X20", False)["passed"]
    assert not answer_score(case, "X20 10:45", True)["passed"]
    with pytest.raises(ValueError):
        ShortAnswer.model_validate_json('{"answer":"X20","abstain":"false"}')


def test_reference_context_never_contains_restricted_rows():
    """未来/外人/失败轮次不会进入完整历史参考；模型不应看到gold标签。"""
    for case in build()["cases"]:
        if case["category"] != "no_answer":
            continue
        text = json.dumps(full_context(case), ensure_ascii=False)
        assert "gold_turn_ids" not in text
        assert not any(secret in text for secret in case["forbidden_answers"])
        assert all(row["created_at"] <= case["question_time"] for row in visible_history(case))


def test_real_sql_retrieval_and_exception_denominator(tmp_path):
    """通过真实SQL归档和BM25检索取回助手原答；丢失结果照样进入固定分母。"""
    from db.memory_repository import MemoryRepository
    case = next(case for case in build()["cases"] if case["category"] == "exact")
    repo = MemoryRepository("sqlite:///" + (tmp_path / "eval.db").as_posix())
    repo.initialize()
    try:
        mapping = seed_case(repo, case)
        result = retrieval_run(repo, case, mapping, None, dict(max_turns=200, candidate_limit=12, result_limit=3, rrf_k=60))
        assert result["methods"]["bm25"]["hit"]
        assert result["methods"]["bm25"]["ids"][0] == case["gold_turn_ids"][0]
        assert planned_filters(case)[0] is not None
        row = {"id": case["id"], "retrieval": result}
        second = dict(case, id="failed-case")
        summary = aggregate([case, second], [row, {"id": "failed-case", "error": "network failed"}])
        assert summary["retrieval"]["bm25"]["hit_at_3"] == .5
        assert summary["errors"] == 1
    finally:
        repo.engine.dispose()


def test_model_queue_is_separate_from_model_latency():
    """人为限速等待不能被报告成模型耗时，三个并行调用必须共用同一节流入口。"""
    import asyncio
    from types import SimpleNamespace
    from scripts.evaluate_history import CountedModel, RequestGate
    events, starts = [], []

    class Model:
        async def ainvoke(self, messages):
            """只验证调用计量，回答内容不用于任何质量评分。"""
            starts.append(asyncio.get_running_loop().time())
            return SimpleNamespace(usage_metadata={"input_tokens": 3, "output_tokens": 2})

    async def scenario():
        """使用标准库事件循环，不给项目新增异步测试插件依赖。"""
        gate = RequestGate(.02)
        await asyncio.gather(*(CountedModel(Model(), events, "answer", gate).ainvoke([]) for _ in range(3)))

    asyncio.run(scenario())
    assert all(after - before >= .015 for before, after in zip(starts, starts[1:]))
    assert len(events) == 3 and sum(event["input_tokens"] for event in events) == 9
    assert max(event["evaluation_queue_seconds"] for event in events) >= .03
