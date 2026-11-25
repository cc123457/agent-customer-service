"""只重算已有真实回答的指标，不重新调用模型，不改写原始运行或旧版基线。"""
import argparse
import hashlib
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
ROOT = Path(__file__).resolve().parents[1]


def rescore(source, dataset_path, output):
    """核对数据指纹和案例完整性后，计算精确率/拒答混淆矩阵，保留原始错误分母。"""
    from evals.history_metrics import validate_dataset
    from scripts.evaluate_history import aggregate, fingerprint
    dataset_bytes = dataset_path.read_bytes()
    old_manifest = json.loads((source / "manifest.json").read_text(encoding="utf-8"))
    if hashlib.sha256(dataset_bytes).hexdigest() != old_manifest["dataset_sha256"]:
        raise ValueError("标签与原运行指纹不同，不能混用")
    all_cases = validate_dataset(json.loads(dataset_bytes))
    selected = set(old_manifest["case_ids"])
    cases = [case for case in all_cases if case["id"] in selected]
    original = (source / "results.jsonl").read_bytes()
    rows = [json.loads(line) for line in original.decode("utf-8").splitlines()]
    if len({row["id"] for row in rows}) != len(rows) or set(row["id"] for row in rows) != selected:
        raise ValueError("原始案例重复或不完整")
    result = {"mode": "rescore_existing_real_answers_no_calls", "source": str(source.resolve()),
              "source_results_sha256": hashlib.sha256(original).hexdigest(), "original_manifest": old_manifest,
              "metrics_sha256": fingerprint([ROOT / "evals/history_metrics.py", ROOT / "scripts/evaluate_history.py", Path(__file__)]),
              "precision_scope": "金标准必要证据轮次；固定K分母与实际返回分母分别报告，非所有主题相关性的人工标签",
              "overall": aggregate(cases, rows),
              "categories": {category: aggregate([case for case in cases if case["category"] == category],
                              [row for row in rows if row["category"] == category]) for category in sorted({case["category"] for case in cases})}}
    output.parent.mkdir(parents=True, exist_ok=True)
    if output.exists():
        raise FileExistsError("不能覆盖已有重评分报告")
    output.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    return result


def main():
    """独立离线入口：从指定运行目录读取结果，写入新的指标报告。"""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--dataset", type=Path, default=ROOT / "evals/history_cases.json")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    result = rescore(args.source, args.dataset, args.output)
    print(json.dumps(result["overall"], ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
