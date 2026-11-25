"""导出盲评网页并汇总真实人工判定；不使用规则或模型自动填写human_correct。"""
import argparse
import hashlib
import json
import sys
from datetime import datetime
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
ROOT = Path(__file__).resolve().parents[1]
FIELDS = ("content_correct", "abstention_correct", "no_forbidden_disclosure")


def digest(data):
    """用内容指纹绑定原始运行与具体回答，避免人评结果错配到重跑后的答案。"""
    return hashlib.sha256(data).hexdigest()


def export_review(source, dataset_path, output, refresh=False):
    """导出随机顺序的方案盲评资料，正确性字段保持空白，映射键单独保存。"""
    from evals.history_metrics import validate_dataset, visible_history
    manifest = json.loads((source / "manifest.json").read_text(encoding="utf-8"))
    dataset_bytes = dataset_path.read_bytes()
    if digest(dataset_bytes) != manifest["dataset_sha256"]:
        raise ValueError("评审标签与原始运行不一致")
    cases = {case["id"]: case for case in validate_dataset(json.loads(dataset_bytes))}
    original = (source / "results.jsonl").read_bytes()
    rows = [json.loads(line) for line in original.decode("utf-8").splitlines()]
    if len(rows) != len({row["id"] for row in rows}) or set(row["id"] for row in rows) != set(manifest["case_ids"]):
        raise ValueError("原运行不完整或有重复")
    key = {"version": "memory-human-review-v1", "source": str(source.resolve()), "source_sha256": digest(original),
           "dataset_sha256": digest(dataset_bytes), "records": {}}
    items = []
    for row in rows:
        case = cases[row["id"]]
        for arm, result in row["answers"].items():
            answer_hash = digest(json.dumps({"raw": result.get("raw"), "answer": result.get("answer"),
                                          "abstain": result.get("abstain")}, ensure_ascii=False, sort_keys=True).encode("utf-8"))
            blind_id = digest((key["source_sha256"] + row["id"] + arm).encode())[:16]
            key["records"][blind_id] = {"case_id": row["id"], "arm": arm, "answer_sha256": answer_hash}
            items.append({"blind_id": blind_id, "answer_sha256": answer_hash, "question": case["question"],
                          "question_time": case["question_time"], "history": visible_history(case, own_only=True, completed_only=True),
                          "answer": result.get("answer", result.get("raw", "[接口失败，无回答]")),
                          "abstain": result.get("abstain"), "expected_groups": case["answer_groups"],
                          "should_abstain": case["should_abstain"], "forbidden_values": case["forbidden_answers"],
                          "gold_turn_ids": case["gold_turn_ids"]})
    items.sort(key=lambda item: item["blind_id"])
    output.mkdir(parents=True, exist_ok=True)
    if (output / "blind_key.json").exists():
        existing = json.loads((output / "blind_key.json").read_text(encoding="utf-8"))
        if not refresh or existing != key:
            raise FileExistsError("请选择新评审目录；只有原始来源完全相同才允许--refresh更新页面")
    payload = {"version": key["version"], "source_sha256": key["source_sha256"], "items": items}
    template = (ROOT / "evals/review_template.html").read_text(encoding="utf-8")
    serialized = json.dumps(payload, ensure_ascii=False).replace("<", "\\u003c")
    (output / "review.html").write_text(template.replace("__REVIEW_DATA__", serialized), encoding="utf-8")
    (output / "blind_key.json").write_text(json.dumps(key, ensure_ascii=False, indent=2), encoding="utf-8")
    pending = {"version": key["version"], "source_sha256": key["source_sha256"], "status": "awaiting_human_review",
               "total_answers": len(items), "reviewed_answers": 0, "human_accuracy": None}
    if not (output / "status.json").exists():
        (output / "status.json").write_text(json.dumps(pending, ensure_ascii=False, indent=2), encoding="utf-8")
    return pending


def import_review(key_path, reviews_path, output):
    """严格校验来源、布尔评分、署名和时间，未评审不计作通过，缺失覆盖率明确保留。"""
    key = json.loads(key_path.read_text(encoding="utf-8"))
    reviews = json.loads(reviews_path.read_text(encoding="utf-8"))
    if reviews.get("version") != key["version"] or reviews.get("source_sha256") != key["source_sha256"]:
        raise ValueError("评审文件不属于这次运行")
    if digest((Path(key["source"]) / "results.jsonl").read_bytes()) != key["source_sha256"]:
        raise ValueError("原始回答已变化，需要重新导出人评资料")
    assessed, seen = [], set()
    for review in reviews["reviews"]:
        blind_id = review.get("blind_id")
        if blind_id not in key["records"] or blind_id in seen:
            raise ValueError("未知或重复评审ID")
        seen.add(blind_id)
        entry = key["records"][blind_id]
        if review.get("answer_sha256") != entry["answer_sha256"]:
            raise ValueError("具体回答指纹不匹配")
        if any(type(review.get(field)) is not bool for field in FIELDS):
            raise ValueError("必须逐项填写布尔评分，空白不能当作否")
        if not isinstance(review.get("reviewer"), str) or not review["reviewer"].strip() or review.get("method") != "human":
            raise ValueError("缺少人工评审署名或方法标识")
        moment = datetime.fromisoformat(review["reviewed_at"].replace("Z", "+00:00"))
        if moment.tzinfo is None:
            raise ValueError("评审时间必须带时区")
        assessed.append({**entry, **review, "human_correct": all(review[field] for field in FIELDS)})
    arms = sorted({entry["arm"] for entry in key["records"].values()})
    result = {"method": "human_entered_assessments", "source_sha256": key["source_sha256"],
              "total_answers": len(key["records"]), "reviewed_answers": len(assessed),
              "coverage": len(assessed) / len(key["records"]) if key["records"] else None,
              "full_dataset_human_accuracy": sum(row["human_correct"] for row in assessed) / len(assessed) if assessed and len(assessed) == len(key["records"]) else None,
              "arms": {}, "adjudications": assessed,
              "scope": "真实填写的人工判定；未完成时只报已评子集，不外推全量。方法标签和署名由评审人填写。"}
    for arm in arms:
        group = [row for row in assessed if row["arm"] == arm]
        total = sum(entry["arm"] == arm for entry in key["records"].values())
        result["arms"][arm] = {"total": total, "reviewed": len(group), "coverage": len(group) / total,
            "reviewed_subset_accuracy": sum(row["human_correct"] for row in group) / len(group) if group else None,
            "content_accuracy": sum(row["content_correct"] for row in group) / len(group) if group else None,
            "abstention_accuracy": sum(row["abstention_correct"] for row in group) / len(group) if group else None,
            "no_forbidden_disclosure_rate": sum(row["no_forbidden_disclosure"] for row in group) / len(group) if group else None}
    if output.exists():
        raise FileExistsError("请使用新汇总文件，保留旧人工评审记录")
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    return result


def main():
    """导出静态网页或导入浏览器导出的JSON；全程离线，不自动产生人工判定。"""
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    export = sub.add_parser("export")
    export.add_argument("--source", type=Path, required=True)
    export.add_argument("--dataset", type=Path, default=ROOT / "evals/history_cases.json")
    export.add_argument("--output", type=Path, required=True)
    export.add_argument("--refresh", action="store_true", help="仅刷新同一来源页面，保留评审ID、浏览器进度和状态文件")
    load = sub.add_parser("import")
    load.add_argument("--key", type=Path, required=True)
    load.add_argument("--reviews", type=Path, required=True)
    load.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    result = export_review(args.source, args.dataset, args.output, args.refresh) if args.command == "export" else import_review(args.key, args.reviews, args.output)
    print(json.dumps({name: value for name, value in result.items() if name != "adjudications"}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
