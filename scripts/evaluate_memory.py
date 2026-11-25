"""固定记忆功能集：明确区分词法、注入向量和真实Embedding模式，不把命中率冒充回答准确率。"""
import argparse
import json
import sys
import tempfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))


def evaluate(repository, cases, semantic_index=None):
    """用真实关系库写入事实并跨会话检索，逐例记录预期命中、错误旧值和应澄清状态。"""
    from memory.long_term import extract_memories, recall_memories
    results = []
    for case in cases:
        writer = case.get("writer", "owner") + ":" + case["id"]
        reader = case.get("reader", case.get("writer", "owner")) + ":" + case["id"]
        scope = writer + ":write"
        assert repository.claim(scope, writer, "write", "eval", 3600)
        for number, text in enumerate(case["writes"]):
            turn_id, _ = repository.start_request(scope, writer, "eval", str(number), text, extract_memories(text))
            repository.finish_request(scope, "eval", turn_id, "记录完成")
        repository.release(scope, "eval")
        if semantic_index:
            semantic_index.sync_user(writer)
        kwargs = {"semantic_index": semantic_index} if semantic_index else {}
        rows = recall_memories(repository, reader, case["query"], [], **kwargs)
        values = [row["value"] for row in rows]
        joined = "\n".join(values)
        matched = sum(value in joined for value in case.get("expected", []))
        wrong = any(value in joined for value in case.get("forbidden", [])) or any(
            value in values for value in case.get("forbidden_exact_values", [])
        )
        clarify = any(row.get("needs_clarification") for row in rows)
        expected_count = len(case.get("expected", []))
        passed = matched == expected_count and not wrong and (not case.get("must_clarify") or clarify)
        results.append({"id": case["id"], "passed": passed, "expected_count": expected_count,
                        "matched_count": matched, "wrong_recall": wrong, "clarification": clarify})
    return {"cases": len(results), "passed": sum(row["passed"] for row in results),
            "expected_facts": sum(row["expected_count"] for row in results),
            "matched_facts": sum(row["matched_count"] for row in results),
            "wrong_recall_cases": sum(row["wrong_recall"] for row in results), "results": results,
            "scope": "固定开发功能集、跨会话记忆输入断言；不是独立测试集或模型最终回答准确率。"}


def main():
    """默认离线词法评测；--semantic显式调用真实Embedding，结果写入被Git忽略的运行目录。"""
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, default=Path("evals/runs/memory-lexical.json"))
    parser.add_argument("--semantic", action="store_true")
    parser.add_argument("--env-file", type=Path)
    args = parser.parse_args()
    if args.env_file:
        from dotenv import load_dotenv
        load_dotenv(args.env_file, override=True)
    from db.memory_repository import MemoryRepository
    cases = json.loads(Path("evals/memory_cases.json").read_text(encoding="utf-8"))
    with tempfile.TemporaryDirectory(prefix="memory-eval-") as directory:
        repository = MemoryRepository("sqlite:///" + (Path(directory) / "memory.db").as_posix())
        repository.initialize()
        try:
            index = None
            if args.semantic:
                from memory.semantic_index import SemanticMemoryIndex
                from model.factory import embed_model
                index = SemanticMemoryIndex(repository, Path(directory) / "chroma", embed_model)
            result = evaluate(repository, cases, index)
            result["mode"] = "real_embedding" if args.semantic else "lexical_no_model"
        finally:
            if index:
                index.close()
            repository.engine.dispose()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps({key: value for key, value in result.items() if key != "results"}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
