"""验收协议测试只验证数据隔离、冻结和统计公式，不作为模型回答质量证据。"""
import hashlib
import json
from pathlib import Path

import pytest

from evals.build_memory_qa_holdout import build
from evals.history_metrics import validate_dataset
from scripts.review_memory_holdout import paired, wilson

ROOT = Path(__file__).resolve().parents[1]


def test_holdout_sources_are_valid_and_disjoint_from_development():
    """原始依据有效、用户/问题不与旧集重叠，并能在完整历史预算中评测全部80题。"""
    from scripts.evaluate_history import full_context
    from memory.budget import message_tokens
    data = build()
    assert json.loads((ROOT / "evals/memory_qa_holdout.json").read_text(encoding="utf-8")) == data
    cases = validate_dataset(data)
    assert len(cases) == 80 and sum(c["should_abstain"] for c in cases) == 25
    for name in ("history_cases.json", "refinement_cases.json", "temporal_boundary_cases.json"):
        old = json.loads((ROOT / "evals" / name).read_text(encoding="utf-8"))["cases"]
        for field in ("id", "user_id", "question"):
            assert not {c[field] for c in cases}.intersection(c[field] for c in old)
    for case in cases:
        assert message_tokens(full_context(case)) <= 32768


def test_protocol_refuses_data_or_source_changes(tmp_path, monkeypatch):
    """冻结后题目或实现改动都应立即拒绝执行，不能悄悄继续生成较高分数。"""
    import scripts.run_memory_holdout as runner
    dataset = tmp_path / "dataset.json"
    dataset.write_text(json.dumps(build(), ensure_ascii=False), encoding="utf-8")
    frozen = tmp_path / "freeze.json"
    protected = ROOT / "memory/budget.py"
    monkeypatch.setattr(runner, "protected_files", lambda: [protected])
    runner.freeze(dataset, frozen)
    assert runner.verify(frozen)["dataset_sha256"] == hashlib.sha256(dataset.read_bytes()).hexdigest()
    record = json.loads(frozen.read_text(encoding="utf-8"))
    record["implementation_sha256"]["memory/budget.py"] = "0" * 64
    frozen.write_text(json.dumps(record), encoding="utf-8")
    with pytest.raises(ValueError, match="源码变化"):
        runner.verify(frozen)
    record["implementation_sha256"]["memory/budget.py"] = runner.digest(protected)
    frozen.write_text(json.dumps(record), encoding="utf-8")
    dataset.write_text("{}", encoding="utf-8")
    with pytest.raises(ValueError, match="题目发生变化"):
        runner.verify(frozen)


def test_paired_comparison_and_intervals_keep_failure_denominators():
    """用人工可算的配对例检验胜负、百分点和区间，而非把规则命中冒充语义准确率。"""
    result = paired([True, True, True, False], [True, False, False, True])
    assert result["memory_wins"] == 2 and result["memory_losses"] == 1
    assert result["both_correct"] == 1 and result["both_wrong"] == 0
    assert result["accuracy_difference_pp"] == 25
    assert result["exact_paired_discordance_p"] == 1
    low, high = wilson(0, 80)
    assert low == pytest.approx(0, abs=1e-12) and 0 < high < .05
    low, high = wilson(80, 80)
    assert .95 < low < 1 and high == pytest.approx(1)
    with pytest.raises(ValueError):
        paired([True], [])
