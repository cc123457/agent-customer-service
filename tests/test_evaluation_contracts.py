import asyncio
import json
from pathlib import Path

from scripts.evaluate_agent import contract_cases


def test_fixed_cases_have_unique_ids_and_real_contract_checks():
    root = Path(__file__).resolve().parents[1]
    cases = json.loads((root / "evals/agent_cases.json").read_text(encoding="utf-8"))
    assert len(cases) == len({case["id"] for case in cases}) == 60
    live = [case for case in cases if case["suite"] == "agent"]
    assert len(live) == 48
    assert all(case["turns"] and case["rubric"] for case in live)
    results = asyncio.run(contract_cases([case for case in cases if case["suite"] == "contract"]))
    assert len(results) == 12
    assert all(result["passed"] and result["calls"] == 0 for result in results)


def test_agent_evaluator_uses_current_sql_service_without_external_calls():
    from scripts.evaluate_agent import agent_cases

    class InputEchoAgent:
        async def execute_stream(self, messages, user_id, callbacks):
            """暴露本轮上下文以验证同一案例多轮保留、不同案例互不污染。"""
            yield {"type": "answer", "content": " | ".join(row["content"] for row in messages)}

    cases = [{"id": "case-a", "user_id": "1001", "turns": ["case-a秘密", "再问"], "category": "memory", "rubric": "隔离"},
             {"id": "case-b", "user_id": "1001", "turns": ["case-b问题"], "category": "memory", "rubric": "隔离"}]
    results = asyncio.run(agent_cases(cases, InputEchoAgent(), concurrency=2))
    assert all(row["completed"] for row in results)
    assert "case-a秘密" in results[0]["answers"][-1]
    assert "case-a秘密" not in results[1]["answers"][-1]
