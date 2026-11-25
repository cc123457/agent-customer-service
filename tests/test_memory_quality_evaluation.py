"""质量评测的指标分母、标签依据、人评边界与原始结果指纹测试。"""
import asyncio
import json
from pathlib import Path

import pytest

from evals.build_quality_cases import build, fact
from evals.history_metrics import abstention_metrics, retrieval_score
from evals.quality_metrics import extraction_score, maximum_matches, summary_score
from scripts.evaluate_memory_quality import extract_case, validate_quality

ROOT = Path(__file__).resolve().parents[1]


def test_quality_dataset_frozen_and_sourced():
    """数据生成可复现，标签有原文依据，摘要长样本确实超过生产单批输入预算。"""
    dataset = validate_quality(build())
    assert dataset == json.loads((ROOT / "evals/quality_cases.json").read_text(encoding="utf-8"))
    assert sum(case["split"] == "test" for case in dataset["extraction"]) == 48
    assert sum(case["split"] == "test" for case in dataset["summary"]) == 24
    assert all(len("".join(row["content"] for row in case["messages"]).encode()) > 16000
               for case in dataset["summary"] if case["category"] == "multi_batch")


def test_precision_denominators_and_abstention_failures():
    """返回1条正确证据时固定P@3是1/3，实际返回精确率是1；缺失回答不能计作正确拒答。"""
    case = {"question_time": "2026-11-24T18:00:00+08:00", "history": [], "user_id": "u", "gold_turn_ids": ["g"]}
    score = retrieval_score(case, ["g"])
    assert score["precision_at_k"] == 1 / 3
    assert score["precision_returned"] == 1
    assert retrieval_score(case, ["g", "g"])["relevant_returned"] == 1
    cases = [{"should_abstain": True}, {"should_abstain": False}, {"should_abstain": True}]
    records = [{"structure_valid": True, "abstain": True}, {"structure_valid": True, "abstain": True}, {}]
    metrics = abstention_metrics(cases, records)
    assert metrics["abstention_precision"] == .5 and metrics["abstention_recall"] == .5
    assert metrics["invalid_or_missing"] == 1 and metrics["over_refusals"] == 1
    assert metrics["abstention_f1"] == .5


def test_extraction_matching_is_one_to_one_and_time_sensitive():
    """重复预测算FP，有效日期错误同时算FP和FN，别名交叉用最大匹配而非贪心。"""
    gold = [fact("device_model", "X20", aliases=["X21"]), fact("device_model", "X20")]
    predictions = [{"key": "device_model", "value": value, "entity": "profile", "action": "upsert", "valid_at": ""} for value in ("X20", "X21")]
    assert len(maximum_matches(predictions, gold)) == 2
    case = {"text": "X20 X21", "expected": gold[:1]}
    score = extraction_score(case, predictions)
    assert score["tp"] == score["fp"] == 1 and score["fn"] == 0
    case["expected"][0]["valid_at"] = "2026-12-01"
    score = extraction_score(case, predictions[:1])
    assert score["tp"] == 0 and score["fp"] == score["fn"] == 1


def test_summary_retention_does_not_hide_known_contradictions():
    """正确关键词出现不抵消同时声明已修复；遗漏与无来源编号分别计量。"""
    case = {"old_summary": "登记号TK-1000", "messages": [{"content": "故障尚未解决。"}],
            "facts": [{"id": "ticket", "groups": [["TK-1000"]]}, {"id": "open", "groups": [["尚未解决"]]}],
            "known_wrong_claims": [{"label": "resolved", "pattern": "故障已解决"}]}
    result = summary_score(case, "登记号TK-1000，故障尚未解决。故障已解决，改用ZZ-9999。")
    assert result["retained"] == 2 and result["known_wrong_claims"] == ["resolved"]
    assert result["unsupported_identifiers"] == ["ZZ-9999"]
    assert not summary_score(case, "不能说故障已解决。登记号TK-1000，尚未解决。")["known_wrong_claims"]


def test_live_path_uses_production_extractor_and_retains_usage():
    """注入模型仅控制测试路径，验证JSON Mode绑定和模型候选真正经过生产验证器。"""
    from types import SimpleNamespace
    from scripts.evaluate_history import RequestGate
    bound = []

    class Model:
        def bind(self, **kwargs):
            """记录提取器是否绑定JSON Mode，绝不将此替身成绩当真实质量。"""
            bound.append(kwargs)
            return self

        async def ainvoke(self, messages):
            """返回合法原文候选以验证计量和校验链路。"""
            content = {"facts": [{"key": "note", "value": "我对夜间噪声敏感", "entity": "profile", "evidence": "我对夜间噪声敏感", "valid_at": "", "action": "upsert"}]}
            return SimpleNamespace(content=json.dumps(content, ensure_ascii=False), usage_metadata={"input_tokens": 7, "output_tokens": 5})

    case = {"id": "test-note", "category": "note", "text": "我对夜间噪声敏感。", "expected": [fact("note", "我对夜间噪声敏感")]}
    result = asyncio.run(extract_case(case, Model(), RequestGate(0)))
    assert result["modes"]["rules_model"]["score"]["case_exact"]
    assert result["modes"]["rules_model"]["calls"][0]["input_tokens"] == 7
    assert bound == [{"response_format": {"type": "json_object"}, "max_tokens": 1000}]


def test_human_review_requires_actual_fields_and_tracks_missing(tmp_path):
    """人评空白不能变成通过；部分评审只报告子集，不伪造全量人工准确率。"""
    from scripts.review_memory import digest, export_review, import_review
    dataset = {"cases": [{"id": "case", "split": "test", "user_id": "owner", "session_id": "s",
               "question_time": "2026-11-24T18:00:00+08:00", "question": "编号？", "history": [
               {"id": "g", "user_id": "owner", "session_id": "s", "created_at": "2026-11-23T10:00:00+08:00", "status": "completed", "question": "编号？", "answer": "TK-1000"}],
               "gold_turn_ids": ["g"], "answer_groups": [["TK-1000"]], "forbidden_answers": [], "should_abstain": False}]}
    data = tmp_path / "data.json"
    data.write_text(json.dumps(dataset), encoding="utf-8")
    source = tmp_path / "source"
    source.mkdir()
    (source / "manifest.json").write_text(json.dumps({"dataset_sha256": digest(data.read_bytes()), "case_ids": ["case"]}), encoding="utf-8")
    answers = {arm: {"answer": "TK-1000", "raw": "{}", "abstain": False} for arm in ("recent", "memory")}
    (source / "results.jsonl").write_text(json.dumps({"id": "case", "answers": answers}) + "\n", encoding="utf-8")
    output = tmp_path / "review"
    pending = export_review(source, data, output)
    assert pending["human_accuracy"] is None and pending["reviewed_answers"] == 0
    html = (output / "review.html").read_text(encoding="utf-8")
    assert "__REVIEW_DATA__" not in html
    assert "gold_turn_ids" in html and 'id="evidenceOnly"' in html
    assert export_review(source, data, output, refresh=True)["reviewed_answers"] == 0
    key = json.loads((output / "blind_key.json").read_text(encoding="utf-8"))
    blind_id, entry = next(iter(key["records"].items()))
    row = {"blind_id": blind_id, "answer_sha256": entry["answer_sha256"], "method": "human", "reviewer": "test fixture", "reviewed_at": "2026-11-24T12:00:00Z", "content_correct": True, "abstention_correct": True, "no_forbidden_disclosure": True}
    review = tmp_path / "reviews.json"
    review.write_text(json.dumps({"version": key["version"], "source_sha256": key["source_sha256"], "reviews": [row]}), encoding="utf-8")
    result = import_review(output / "blind_key.json", review, tmp_path / "scores.json")
    assert result["coverage"] == .5 and result["full_dataset_human_accuracy"] is None
    row["content_correct"] = None
    review.write_text(json.dumps({"version": key["version"], "source_sha256": key["source_sha256"], "reviews": [row]}), encoding="utf-8")
    with pytest.raises(ValueError, match="布尔评分"):
        import_review(output / "blind_key.json", review, tmp_path / "bad.json")


def test_rescore_quality_preserves_frozen_predictions(tmp_path):
    """原始分数经JSON保存后再次计算仍一致，不能把tuple/list序列化差异误报成评分变化。"""
    from scripts.rescore_quality import rescore_quality
    from scripts.review_memory import digest
    case = {"id": "test-item-00", "category": "device", "split": "test", "text": "我用的是X20。", "expected": [fact("device_model", "X20")]}
    data = tmp_path / "dataset.json"
    data.write_text(json.dumps({"extraction": [case], "summary": []}), encoding="utf-8")
    predictions = [{"key": "device_model", "value": "X20", "entity": "profile", "action": "upsert", "valid_at": "", "evidence": "我用的是X20。"}]
    row = {"id": case["id"], "suite": "extraction", "category": "device", "modes": {"rules": {
          "predictions": predictions, "score": extraction_score(case, predictions), "calls": [], "error": None}}}
    source = tmp_path / "run"
    source.mkdir()
    (source / "manifest.json").write_text(json.dumps({"dataset_sha256": digest(data.read_bytes()), "case_ids": ["extraction:" + case["id"]]}), encoding="utf-8")
    (source / "results.jsonl").write_text(json.dumps(row) + "\n", encoding="utf-8")
    result = rescore_quality(source, data, tmp_path / "rescored.json")
    assert result["changed_score_cases"] == []
    assert result["overall"]["extraction"]["rules"]["tp"] == 1
