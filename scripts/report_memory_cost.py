"""从真实评测usage计算同一负载的参考费用；免费额度和实际扣款不由Token数量推断。"""
import argparse
import hashlib
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from model.usage import combine_reports


def report_cost(source, free_quota_reported=False):
    """仅当前方案实际建索引/摘要/查询/回答计入当前成本，排除独立诊断和其他回答方案。"""
    original = (source / "results.jsonl").read_bytes()
    rows = [json.loads(line) for line in original.decode("utf-8").splitlines()]
    manifest = json.loads((source / "manifest.json").read_text(encoding="utf-8"))
    if len({row["id"] for row in rows}) != len(rows) or set(row["id"] for row in rows) != set(manifest["case_ids"]):
        raise ValueError("评测案例缺失或重复")
    if any("usage_ledger" not in row for row in rows):
        raise ValueError("原运行没有Embedding末端usage，不能按字节数补造费用")
    result = combine_reports(row["usage_ledger"] for row in rows)
    stages = result["stages"]
    memory = stages.get("memory", {})
    full = stages.get("full_history", {})
    summary = stages.get("summary", {})
    full_input = full.get("input_tokens_reported", 0)
    memory_input = memory.get("input_tokens_reported", 0)
    summary_input = summary.get("input_tokens_reported", 0)
    sufficient = (memory.get("calls", 0) >= len(rows) and full.get("calls", 0) >= len(rows)
                  and all(not row.get("error") for row in rows) and result["comparison_usage_complete"])
    result.update(source=str(source.resolve()), source_results_sha256=hashlib.sha256(original).hexdigest(),
        free_quota_reported_by_user=free_quota_reported, actual_bill_paid=None,
        answer_input_token_reduction=1 - memory_input / full_input if full_input and sufficient else None,
        answer_plus_summary_input_token_reduction=1 - (memory_input + summary_input) / full_input if full_input and sufficient else None,
        usage_comparison_valid=sufficient,
        memory_embedding_reference_cost=sum(stages.get(stage, {}).get("reference_cost_known_subtotal", 0)
                                          for stage in ("memory_context/documents", "memory_context/query")),
        diagnostics_reference_cost=sum(value["reference_cost_known_subtotal"] for name, value in stages.items()
                                       if name.startswith(("diagnostic_", "history_index_build/"))))
    if not sufficient:
        result["cold_reference_cost_reduction"] = None
    return result


def main():
    """离线读取固定运行写新报告，不发起模型调用，也不覆盖已有账本。"""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--free-quota-reported", action="store_true")
    args = parser.parse_args()
    result = report_cost(args.source, args.free_quota_reported)
    if args.output.exists():
        raise FileExistsError("请选择新费用报告文件，保留旧测量")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps({key: value for key, value in result.items() if key != "stages"}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
