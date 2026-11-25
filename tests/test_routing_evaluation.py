"""评测基线必须先能构造有效输入，不能将评测器异常计成模型能力低。"""
import json
from pathlib import Path

from memory.budget import message_tokens
from scripts.evaluate_memory_routing import recent_context


def test_all_recent_baselines_have_turn_ids_and_preserve_question():
    """遍历固定40题但不调用模型，验证问答配对、窗口和预算契约。"""
    dataset=json.loads(Path("evals/memory_routing_holdout.json").read_text(encoding="utf-8"))
    for case in dataset["cases"]:
        context=recent_context(case,dict(history_window=20,context_budget=8192))
        assert context[-1]==dict(role="user",content=case["question"])
        assert message_tokens(context)<=8192
        assert all(context[index]["role"]!=context[index+1]["role"] for index in range(len(context)-1))
