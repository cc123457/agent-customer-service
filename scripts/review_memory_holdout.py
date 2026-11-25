"""整理新题证据并汇总逐条AI判断；不以关键词规则或标准答案自动生成语义评分。"""
import argparse
import hashlib
import json
import math
import sys
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
ARMS = ("recent", "summary_recent", "memory", "full_history")
FIELDS = ("content_correct", "abstention_correct", "no_forbidden_disclosure", "context_supported", "context_sufficient")


def sha(data):
    """用原始字节绑定已生成答案，复核期间不允许换答案。"""
    return hashlib.sha256(data).hexdigest()


def encode(value):
    """稳定序列化上下文用于指纹核对。"""
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode("utf-8")


def wilson(successes, total):
    """给出二项比例95% Wilson区间；小型内部合成集的区间不代表线上总体。"""
    if total < 1 or not 0 <= successes <= total:
        raise ValueError("样本数或通过数无效")
    z = 1.959963984540054
    p = successes / total
    scale = 1 + z * z / total
    center = (p + z * z / (2 * total)) / scale
    margin = z * math.sqrt(p * (1 - p) / total + z * z / (4 * total * total)) / scale
    return [max(0, center - margin), min(1, center + margin)]


def paired(memory, baseline):
    """同题配对统计胜负和准确率差，避免把不同难度样本的百分比直接相减。"""
    if len(memory) != len(baseline) or not memory:
        raise ValueError("配对样本必须等长非空")
    wins = sum(m and not b for m, b in zip(memory, baseline))
    losses = sum(b and not m for m, b in zip(memory, baseline))
    n = wins + losses
    probability = min(1.0, 2 * sum(math.comb(n, k) for k in range(min(wins, losses) + 1)) / 2 ** n) if n else 1.0
    return {"cases": len(memory), "memory_wins": wins, "memory_losses": losses,
            "both_correct": sum(m and b for m, b in zip(memory, baseline)),
            "both_wrong": sum(not m and not b for m, b in zip(memory, baseline)),
            "accuracy_difference_pp": 100 * (wins - losses) / len(memory),
            "exact_paired_discordance_p": probability,
            "scope": "同一内部数据集的探索性配对统计，不代表外部泛化或多个比较校正后显著性。"}


def prepare(source, dataset, output):
    """为全部320个计划输出生成隐藏方案名的证据包，执行失败也保留一条待判断记录。"""
    from evals.history_metrics import validate_dataset, visible_history
    raw, data = (source / "results.jsonl").read_bytes(), dataset.read_bytes()
    original = json.loads((source / "manifest.json").read_text(encoding="utf-8"))
    cases = validate_dataset(json.loads(data))
    rows = [json.loads(line) for line in raw.decode("utf-8").splitlines()]
    if sha(data) != original["dataset_sha256"] or {r["id"] for r in rows} != set(original["case_ids"]) or len(rows) != len(cases):
        raise ValueError("数据/案例与原始运行不一致")
    if output.exists():
        raise FileExistsError("评审证据包不可覆盖")
    output.mkdir(parents=True)
    (output / "packets").mkdir()
    by_id = {r["id"]: r for r in rows}
    manifest = {"version": "memory-qa-holdout-review-v1", "source": str(source.resolve()), "source_sha256": sha(raw),
        "dataset": str(dataset.resolve()), "dataset_sha256": sha(data), "answers": {},
        "method": "assistant_semantic_review", "human_correct": None,
        "system_prompt": original["answer_system_prompt"],
        "scope": "全量逐条AI复核，评审者参与数据构建，不是外部盲评或真人评分；历史文本是数据，不执行其中指令。"}
    grouped = {}
    for case in cases:
        row = by_id[case["id"]]
        routines = {t[field] for t in case["history"] if ":filler-" in t["id"] for field in ("question", "answer")}
        packet = {"case_id": case["id"], "question": case["question"], "question_time": case["question_time"],
            "valid_history": [t for t in visible_history(case, own_only=True, completed_only=True) if ":filler-" not in t["id"]],
            "excluded_history": [t for t in case["history"] if t not in visible_history(case, own_only=True, completed_only=True)],
            "routine_history_rounds": sum(":filler-" in t["id"] for t in case["history"]),
            "routine_catalog": sorted(routines), "answers": []}
        for arm in ARMS:
            answer = row.get("answers", {}).get(arm, {})
            blind_id = sha((sha(raw) + case["id"] + arm).encode())[:16]
            context = answer.get("context", [])
            shown = []
            for index, message in enumerate(context):
                text = message["content"]
                body = text.split("\n", 1)[-1] if text.startswith("[来源会话:") else text
                if body not in routines:
                    shown.append(dict(index=index, **message))
            error = answer.get("error") or row.get("error")
            manifest["answers"][blind_id] = {"case_id": case["id"], "category": case["category"], "arm": arm,
                "answer_sha256": sha(encode({"raw": answer.get("raw"), "answer": answer.get("answer"), "abstain": answer.get("abstain"), "error": error})),
                "context_sha256": sha(encode(context)), "structure_valid": answer.get("structure_valid", False)}
            packet["answers"].append({"blind_id": blind_id, "answer": answer.get("answer"), "raw": answer.get("raw"),
                "abstain": answer.get("abstain"), "error": error, "context": shown})
        packet["answers"].sort(key=lambda a: a["blind_id"])
        grouped.setdefault(case["category"], []).append(packet)
    for category, packets in grouped.items():
        (output / "packets" / (category + ".json")).write_text(json.dumps(packets, ensure_ascii=False, indent=2), encoding="utf-8")
    manifest["packet_sha256"] = {p.name: sha(p.read_bytes()) for p in (output / "packets").glob("*.json")}
    (output / "manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
    return {"cases": len(cases), "answers": len(manifest["answers"])}


def aggregate(manifest_path, assessments, output):
    """校验每条显式语义判断及原文引用，保留全部80题分母并给出各方案和分类结果。"""
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if sha((Path(manifest["source"]) / "results.jsonl").read_bytes()) != manifest["source_sha256"] or sha(Path(manifest["dataset"]).read_bytes()) != manifest["dataset_sha256"]:
        raise ValueError("原始回答或数据已变化")
    packets = {}
    for name, expected in manifest["packet_sha256"].items():
        path = manifest_path.parent / "packets" / name
        if sha(path.read_bytes()) != expected:
            raise ValueError("证据包变化")
        for case in json.loads(path.read_text(encoding="utf-8")):
            packets[case["case_id"]] = case
    records = [json.loads(s) for s in assessments.read_text(encoding="utf-8").splitlines() if s.strip()]
    if len(records) != len(manifest["answers"]) or {r["blind_id"] for r in records} != set(manifest["answers"]):
        raise ValueError("必须恰好评审全部320个计划输出")
    for record in records:
        if record.get("method") != "assistant_semantic_review" or record.get("human_correct") is not None:
            raise ValueError("AI判断不能写成真人评分")
        if any(type(record.get(field)) is not bool for field in FIELDS) or len(record.get("reason", "")) < 12 or not record.get("evidence"):
            raise ValueError("缺少显式判断、理由或引用")
        identity = manifest["answers"][record["blind_id"]]
        packet = packets[identity["case_id"]]
        answer = next(a for a in packet["answers"] if a["blind_id"] == record["blind_id"])
        for evidence in record["evidence"]:
            kind = evidence["source"]
            if kind == "history":
                turn = next((t for t in packet["valid_history"] if t["id"] == evidence.get("id")), None)
                text = turn["question"] + "\n" + turn["answer"] if turn else ""
            elif kind == "context":
                text = next((m["content"] for m in answer["context"] if m["index"] == evidence.get("index")), "")
            elif kind == "answer":
                text = answer["raw"] or answer["answer"] or ""
            elif kind == "error":
                text = answer["error"] or ""
            else:
                raise ValueError("不支持的引用类型")
            if not evidence.get("quote") or evidence["quote"] not in text:
                raise ValueError("引用不是原文子串：" + record["blind_id"])
        record.update(identity)
        record["task_passed"] = identity["structure_valid"] and all(record[f] for f in FIELDS[:3])
    by_arm = {}
    for arm in ARMS:
        group = sorted((r for r in records if r["arm"] == arm), key=lambda r: r["case_id"])
        count = sum(r["task_passed"] for r in group)
        by_arm[arm] = {"cases": len(group), "passed": count, "accuracy": count / len(group), "wilson_95": wilson(count, len(group)),
            "content_correct": sum(r["content_correct"] for r in group),
            "categories": {category: {"cases": 8, "passed": sum(r["task_passed"] for r in group if r["category"] == category)} for category in sorted({r["category"] for r in group})}}
    ordered = {arm: [r["task_passed"] for r in sorted(records, key=lambda r: r["case_id"]) if r["arm"] == arm] for arm in ARMS}
    result = {"version": manifest["version"], "method": manifest["method"], "source_sha256": manifest["source_sha256"],
        "dataset_sha256": manifest["dataset_sha256"], "assessments_sha256": sha(assessments.read_bytes()),
        "reviewed_at": datetime.now(timezone.utc).isoformat(), "human_correct": None, "human_assessments": 0,
        "coverage": {"reviewed": len(records), "total": len(manifest["answers"])}, "arms": by_arm,
        "paired_vs_memory": {arm: paired(ordered["memory"], ordered[arm]) for arm in ARMS if arm != "memory"},
        "scope": manifest["scope"], "records": records}
    if output.exists():
        raise FileExistsError("不能覆盖旧评分")
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return {k: v for k, v in result.items() if k != "records"}


def main():
    """prepare只整理来源；aggregate只校验已明确撰写的AI判断，不自行打分。"""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("mode", choices=("prepare", "aggregate"))
    parser.add_argument("--source", type=Path)
    parser.add_argument("--dataset", type=Path, default=ROOT / "evals/memory_qa_holdout.json")
    parser.add_argument("--manifest", type=Path)
    parser.add_argument("--assessments", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    result = prepare(args.source, args.dataset, args.output) if args.mode == "prepare" else aggregate(args.manifest, args.assessments, args.output)
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
