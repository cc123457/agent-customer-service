"""AI复核的完整性与证据校验测试；测试夹具的布尔值不代表真实模型成绩。"""
import copy
import json

import pytest

from scripts.review_memory_ai import aggregate_review, prepare, render_review, sha


@pytest.fixture
def review_fixture(tmp_path):
    """构造四条最小回答，独立于真实192条记录，不产生人工评审。"""
    dataset = {"cases": [{"id": "test-fixture-00", "split": "test", "category": "exact", "user_id": "owner", "session_id": "s",
               "question_time": "2026-11-24T18:00:00+08:00", "question": "编号是什么？", "history": [
               {"id": "test-fixture-00:g", "user_id": "owner", "session_id": "s", "created_at": "2026-11-23T10:00:00+08:00",
                "status": "completed", "question": "请记录编号", "answer": "登记号为TK-1000。"}],
               "gold_turn_ids": ["test-fixture-00:g"], "answer_groups": [["TK-1000"]], "forbidden_answers": [], "should_abstain": False}]}
    path = tmp_path / "dataset.json"
    path.write_text(json.dumps(dataset, ensure_ascii=False), encoding="utf-8")
    source = tmp_path / "source"
    source.mkdir()
    (source / "manifest.json").write_text(json.dumps({"dataset_sha256": sha(path.read_bytes()), "case_ids": ["test-fixture-00"]}), encoding="utf-8")
    answers = {arm: {"answer": "TK-1000", "raw": '{"answer":"TK-1000","abstain":false}', "abstain": False,
                    "context": [{"role": "assistant", "content": "登记号为TK-1000。"}, {"role": "user", "content": "编号是什么？"}]}
               for arm in ("recent", "summary_recent", "memory", "full_history")}
    (source / "results.jsonl").write_text(json.dumps({"id": "test-fixture-00", "answers": answers}, ensure_ascii=False) + "\n", encoding="utf-8")
    output = tmp_path / "review"
    prepare(source, path, output)
    manifest = json.loads((output / "manifest.json").read_text(encoding="utf-8"))
    records = [{"blind_id": blind_id, "method": "assistant_semantic_review", "human_correct": None,
                "task_content_correct": True, "task_abstention_correct": True, "context_evidence_sufficient": True,
                "context_response_supported": True, "context_abstention_correct": True, "no_forbidden_disclosure": True,
                "needs_human_review": False, "context_behavior": "supported_answer", "issue_codes": [], "confidence": "high",
                "reason": "测试夹具：原文和输入都包含登记号，回答可被所提供的证据支持。", "evidence": [
                {"source": "history", "id": "test-fixture-00:g", "quote": "登记号为TK-1000。"},
                {"source": "context", "index": 0, "quote": "登记号为TK-1000。"}]}
               for blind_id in manifest["answers"]]
    return output, source, records


def write_records(output, records):
    """写入独立测试评审文件，避免将其混入真实人工或AI结果。"""
    path = output / "fixture-records.jsonl"
    path.write_text("".join(json.dumps(record, ensure_ascii=False) + "\n" for record in records), encoding="utf-8")
    return path


def test_ai_review_separates_task_and_context(review_fixture):
    """任务未通过和上下文行为正确可以同时成立，AI结果绝不能变成人工评分。"""
    output, _, records = review_fixture
    records[0].update(task_content_correct=False, task_abstention_correct=False, context_evidence_sufficient=False,
                      context_behavior="justified_abstention", issue_codes=["task_evidence_missing"])
    result = aggregate_review(output / "manifest.json", write_records(output, records), output / "summary.json")
    assert result["coverage"] == {"reviewed": 4, "total": 4}
    assert result["human_assessments"] == 0 and result["human_correct"] is None
    assert sum(arm["ai_task_passed"] for arm in result["arms"].values()) == 3
    assert sum(arm["context_response_supported"] for arm in result["arms"].values()) == 4
    render_review(output / "summary.json", output / "manifest.json", output / "review.html")
    html = (output / "review.html").read_text(encoding="utf-8")
    assert "__AI_REVIEW_DATA__" not in html
    assert "完整历史上的任务结果" in html and "基于模型当时实际输入" in html


@pytest.mark.parametrize("defect", ["missing", "duplicate", "invented_quote", "human_score", "missing_boolean"])
def test_ai_review_rejects_invalid_assessments(review_fixture, defect):
    """漏评、重评、编造引句、人评伪装和缺少明确判定都必须被拒绝。"""
    output, _, original = review_fixture
    records = copy.deepcopy(original)
    if defect == "missing":
        records.pop()
    elif defect == "duplicate":
        records.append(records[0])
    elif defect == "invented_quote":
        records[0]["evidence"][0]["quote"] = "不存在的证据TK-9999"
    elif defect == "human_score":
        records[0]["human_correct"] = True
    else:
        records[0].pop("context_abstention_correct")
    with pytest.raises(ValueError):
        aggregate_review(output / "manifest.json", write_records(output, records), output / "invalid.json")


def test_ai_review_rejects_changed_source(review_fixture):
    """评审绑定原始输出字节，重跑或修改回答后不能复用旧判定。"""
    output, source, records = review_fixture
    with (source / "results.jsonl").open("a", encoding="utf-8") as stream:
        stream.write("\n")
    with pytest.raises(ValueError, match="原始回答已改变"):
        aggregate_review(output / "manifest.json", write_records(output, records), output / "changed.json")
