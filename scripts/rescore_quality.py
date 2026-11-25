"""离线重新验证抽取/摘要标签和评分器，不重跑模型，不覆盖原始结果。"""
import argparse
import copy
import hashlib
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
ROOT = Path(__file__).resolve().parents[1]


def rescore_quality(source, dataset, output):
    """固定原始预测及摘要，只更新确定性指标并记录差异和新旧指纹。"""
    from scripts.evaluate_memory_quality import aggregate_quality, validate_quality
    from scripts.evaluate_history import fingerprint
    from evals.quality_metrics import extraction_score, summary_score
    manifest = json.loads((source / "manifest.json").read_text(encoding="utf-8"))
    data_bytes = dataset.read_bytes()
    if hashlib.sha256(data_bytes).hexdigest() != manifest["dataset_sha256"]:
        raise ValueError("数据标签指纹与原运行不同")
    data = validate_quality(json.loads(data_bytes))
    cases = {suite + ":" + case["id"]: case for suite in ("extraction", "summary") for case in data[suite]}
    raw_bytes = (source / "results.jsonl").read_bytes()
    rows = [json.loads(line) for line in raw_bytes.decode("utf-8").splitlines()]
    identifiers = [row["suite"] + ":" + row["id"] for row in rows]
    if len(identifiers) != len(set(identifiers)) or set(identifiers) != set(manifest["case_ids"]):
        raise ValueError("原始运行不完整或重复")
    originals, changes = copy.deepcopy(rows), []
    for row, original in zip(rows, originals, strict=True):
        case = cases[row["suite"] + ":" + row["id"]]
        if row["suite"] == "extraction":
            for record in row["modes"].values():
                record["score"] = extraction_score(case, record["predictions"])
        else:
            row["score"] = summary_score(case, row["summary"])
        if row != original:
            changes.append(row["suite"] + ":" + row["id"])
    result = {"mode": "offline_rescore_no_model_calls", "source": str(source.resolve()),
              "source_results_sha256": hashlib.sha256(raw_bytes).hexdigest(), "original_manifest": manifest,
              "metrics_sha256": fingerprint([ROOT / "evals/quality_metrics.py", ROOT / "scripts/evaluate_memory_quality.py", Path(__file__)]),
              "changed_score_cases": changes, "overall": aggregate_quality(rows),
              "categories": {category: aggregate_quality([row for row in rows if row["category"] == category])
                             for category in sorted({row["category"] for row in rows})}}
    if output.exists():
        raise FileExistsError("请使用新文件名保留旧结果")
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    return result


def main():
    """明确选择原运行，默认使用固定质量集；不会调用模型或写业务库。"""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--dataset", type=Path, default=ROOT / "evals/quality_cases.json")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    result = rescore_quality(args.source, args.dataset, args.output)
    print(json.dumps({"changed_score_cases": result["changed_score_cases"], "overall": result["overall"]}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
