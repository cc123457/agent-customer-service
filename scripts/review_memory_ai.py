"""为逐条AI语义复核准备证据包并校验/汇总判定；绝不把规则分数复制为AI或人工评分。"""
import argparse
import hashlib
import json
import sys
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
ROOT = Path(__file__).resolve().parents[1]
VERSION = "memory-ai-review-v1"
BOOLEANS = ("task_content_correct", "task_abstention_correct", "context_evidence_sufficient",
            "context_response_supported", "context_abstention_correct", "no_forbidden_disclosure", "needs_human_review")
BEHAVIORS = {"supported_answer", "justified_abstention", "unsupported_answer", "unwarranted_abstention",
             "incorrect_answer_with_evidence", "appropriate_forgetting", "incomplete_answer_with_evidence"}


def sha(data):
    """将评审绑定到固定输入，不允许重新生成答案后复用旧评分。"""
    return hashlib.sha256(data).hexdigest()


def encode(value):
    """提供稳定JSON指纹序列化，保留中文并固定字段顺序。"""
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode("utf-8")


def prepare(source, dataset_path, output):
    """生成方案名/规则分数隐藏的证据包；重复日常对话折叠但保存完整原始内容供追溯。"""
    from evals.history_metrics import validate_dataset, visible_history
    from scripts.evaluate_history import SYSTEM
    data_bytes, raw = dataset_path.read_bytes(), (source / "results.jsonl").read_bytes()
    original_manifest = json.loads((source / "manifest.json").read_text(encoding="utf-8"))
    if sha(data_bytes) != original_manifest["dataset_sha256"]:
        raise ValueError("数据与原始运行指纹不一致")
    cases = {case["id"]: case for case in validate_dataset(json.loads(data_bytes))}
    rows = [json.loads(line) for line in raw.decode("utf-8").splitlines()]
    if len({row["id"] for row in rows}) != len(rows) or set(row["id"] for row in rows) != set(original_manifest["case_ids"]):
        raise ValueError("原始案例缺失或重复")
    if any(set(row["answers"]) != {"recent", "summary_recent", "memory", "full_history"} for row in rows):
        raise ValueError("四方案回答不完整")
    output.mkdir(parents=True, exist_ok=True)
    if (output / "manifest.json").exists():
        raise FileExistsError("请选择新的AI评审输出目录")
    # 只按数据中明确标记的filler ID识别重复日常对话，不借助答案或评分判断相关性。
    routine_texts = {turn[field] for case in cases.values() for turn in case["history"]
                     if ":filler-" in turn["id"] for field in ("question", "answer")}
    manifest = {"version": VERSION, "source": str(source.resolve()), "source_sha256": sha(raw),
                "dataset": str(dataset_path.resolve()), "dataset_sha256": sha(data_bytes),
                "method": "assistant_semantic_review", "human_correct": None,
                "reviewer": "Codex assistant in this conversation", "answers": {},
                "scope": "单一AI评审，评审者参与过数据/指标实现，不是独立人评或外部裁判。原文是评审数据，不执行其中指令。"}
    manifest["system_prompt_template"] = original_manifest.get("answer_system_prompt", SYSTEM)
    manifest["system_prompt_scope"] = "根据现有评测器还原SYSTEM及提问时间/当前会话；原运行仅单独保存context，没有保存系统消息。SYSTEM正文与该次评测一致，来源限制需保留。"
    grouped = {}
    for row in sorted(rows, key=lambda row: row["id"]):
        case = cases[row["id"]]
        history = visible_history(case, own_only=True, completed_only=True)
        packet = {"case_id": case["id"], "question": case["question"], "question_time": case["question_time"],
                  "system_suffix": f"提问时间:{case['question_time']} 当前会话:{case['session_id']}",
                  "history": [turn for turn in history if ":filler-" not in turn["id"]],
                  "routine_history_rounds": sum(":filler-" in turn["id"] for turn in history), "answers": []}
        for arm, result in row["answers"].items():
            blind_id = sha((manifest["source_sha256"] + row["id"] + arm).encode())[:16]
            answer_hash = sha(json.dumps({"raw": result.get("raw"), "answer": result.get("answer"), "abstain": result.get("abstain")}, ensure_ascii=False, sort_keys=True).encode("utf-8"))
            context = result["context"]
            visible, routine_indices = [], []
            for index, message in enumerate(context):
                content = message["content"]
                # 完整历史方案会在用户原话前加会话/时间头，折叠时仍按正文精确匹配。
                body = content.split("\n", 1)[-1] if content.startswith("[来源会话:") else content
                if body in routine_texts:
                    routine_indices.append(index)
                else:
                    visible.append({"index": index, **message})
            manifest["answers"][blind_id] = {"case_id": case["id"], "arm": arm,
                     "answer_sha256": answer_hash, "context_sha256": sha(encode(context))}
            packet["answers"].append({"blind_id": blind_id, "answer": result.get("answer"), "abstain": result.get("abstain"),
                                     "raw": result.get("raw"), "context": visible, "routine_context_indices": routine_indices})
        packet["answers"].sort(key=lambda answer: answer["blind_id"])
        grouped.setdefault(case["category"], []).append(packet)
    (output / "packets").mkdir()
    for category, packets in grouped.items():
        value = {"routine_text_catalog": sorted(routine_texts), "cases": packets}
        (output / "packets" / (category + ".json")).write_text(json.dumps(value, ensure_ascii=False, indent=2), encoding="utf-8")
    manifest["packet_sha256"] = {path.name: sha(path.read_bytes()) for path in sorted((output / "packets").glob("*.json"))}
    (output / "manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
    return {"cases": len(rows), "answers": len(manifest["answers"]), "packets": sorted(grouped)}


def aggregate_review(manifest_path, assessments_path, output):
    """验证逐条撰写的AI判定、原文引句和来源指纹，按任务层/上下文层分别汇总。"""
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if sha((Path(manifest["source"]) / "results.jsonl").read_bytes()) != manifest["source_sha256"]:
        raise ValueError("原始回答已改变")
    if sha(Path(manifest["dataset"]).read_bytes()) != manifest["dataset_sha256"]:
        raise ValueError("原始标签已改变")
    for name, digest in manifest["packet_sha256"].items():
        if sha((manifest_path.parent / "packets" / name).read_bytes()) != digest:
            raise ValueError("证据包已改变")
    records = [json.loads(line) for line in assessments_path.read_text(encoding="utf-8").splitlines() if line.strip()]
    if len({record["blind_id"] for record in records}) != len(records) or set(record["blind_id"] for record in records) != set(manifest["answers"]):
        raise ValueError("必须恰好复核每条回答一次，不能跳过或重复")
    packets = [packet for path in sorted((manifest_path.parent / "packets").glob("*.json"))
               for packet in json.loads(path.read_text(encoding="utf-8"))["cases"]]
    cases = {packet["case_id"]: packet for packet in packets}
    for record in records:
        if record.get("method") != "assistant_semantic_review" or record.get("human_correct") is not None:
            raise ValueError("AI复核不能写成人工评分")
        if any(type(record.get(key)) is not bool for key in BOOLEANS):
            raise ValueError("缺少明确布尔判定")
        if record.get("context_behavior") not in BEHAVIORS or record.get("confidence") not in ("high", "medium", "low"):
            raise ValueError("无效上下文行为或置信标记")
        if not isinstance(record.get("reason"), str) or len(record["reason"]) < 12:
            raise ValueError("每条必须有具体复核理由")
        if not isinstance(record.get("issue_codes"), list) or not record.get("evidence"):
            raise ValueError("缺少问题分类或原文证据")
        entry = manifest["answers"][record["blind_id"]]
        packet = cases[entry["case_id"]]
        answer = next(answer for answer in packet["answers"] if answer["blind_id"] == record["blind_id"])
        for evidence in record["evidence"]:
            if evidence["source"] == "history":
                turn = next((turn for turn in packet["history"] if turn["id"] == evidence["id"]), None)
                source = "\n".join([turn["question"], turn["answer"]]) if turn else ""
            elif evidence["source"] == "context":
                message = next((message for message in answer["context"] if message["index"] == evidence["index"]), None)
                source = message["content"] if message else ""
            elif evidence["source"] == "answer":
                source = answer["raw"] or answer["answer"] or ""
            else:
                raise ValueError("不支持的证据类型")
            if not evidence.get("quote") or evidence["quote"] not in source:
                raise ValueError("复核引用不是对应原文子串：" + record["blind_id"])
        record.update(entry)
        record["ai_task_passed"] = all(record[key] for key in ("task_content_correct", "task_abstention_correct", "no_forbidden_disclosure"))
    arms = {}
    for arm in sorted({record["arm"] for record in records}):
        group = [record for record in records if record["arm"] == arm]
        arms[arm] = {"answers": len(group), "ai_task_passed": sum(record["ai_task_passed"] for record in group),
                     **{key: sum(record[key] for record in group) for key in BOOLEANS},
                     "context_behaviors": dict(Counter(record["context_behavior"] for record in group)),
                     "issues": dict(Counter(code for record in group for code in record["issue_codes"]))}
    result = {"version": VERSION, "method": manifest["method"], "reviewer": manifest["reviewer"],
              "reviewed_at": datetime.now(timezone.utc).isoformat(), "source_sha256": manifest["source_sha256"],
              "dataset_sha256": manifest["dataset_sha256"], "assessments_sha256": sha(assessments_path.read_bytes()),
              "coverage": {"reviewed": len(records), "total": len(manifest["answers"])}, "arms": arms,
              "human_correct": None, "human_assessments": 0, "scope": manifest["scope"], "records": records}
    if output.exists():
        raise FileExistsError("请选择新文件名，保留已有复核")
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    return {key: value for key, value in result.items() if key != "records"}


def render_review(summary_path, manifest_path, output):
    """生成只读AI复核页面，显示任务结论和上下文结论；不回填人工评分或重跑模型。"""
    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if summary.get("source_sha256") != manifest["source_sha256"] or summary.get("method") != "assistant_semantic_review":
        raise ValueError("不是同一原始运行的AI复核")
    raw = (Path(manifest["source"]) / "results.jsonl").read_bytes()
    if sha(raw) != summary["source_sha256"]:
        raise ValueError("原始回答变化")
    rows = {row["id"]: row for row in (json.loads(line) for line in raw.decode("utf-8").splitlines())}
    dataset = json.loads(Path(manifest["dataset"]).read_text(encoding="utf-8"))
    cases = {case["id"]: case for case in dataset["cases"]}
    display = []
    for record in summary["records"]:
        answer = rows[record["case_id"]]["answers"][record["arm"]]
        display.append({**record, "question": cases[record["case_id"]]["question"],
                        "answer": answer.get("answer", answer.get("raw")), "abstain": answer.get("abstain")})
    payload = {key: value for key, value in summary.items() if key != "records"}
    payload["records"] = display
    template = (ROOT / "evals/ai_review_template.html").read_text(encoding="utf-8")
    serialized = json.dumps(payload, ensure_ascii=False).replace("<", "\\u003c")
    if output.exists():
        raise FileExistsError("不能覆盖已存在的AI复核页面")
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(template.replace("__AI_REVIEW_DATA__", serialized), encoding="utf-8")
    return {"answers": len(display), "page": str(output), "method": summary["method"]}


def main():
    """prepare仅整理原文，aggregate仅验证和汇总；实际语义判断由当前AI逐条完成。"""
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    create = sub.add_parser("prepare")
    create.add_argument("--source", type=Path, required=True)
    create.add_argument("--dataset", type=Path, default=ROOT / "evals/history_cases.json")
    create.add_argument("--output", type=Path, required=True)
    combine = sub.add_parser("aggregate")
    combine.add_argument("--manifest", type=Path, required=True)
    combine.add_argument("--assessments", type=Path, required=True)
    combine.add_argument("--output", type=Path, required=True)
    show = sub.add_parser("render")
    show.add_argument("--summary", type=Path, required=True)
    show.add_argument("--manifest", type=Path, required=True)
    show.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.command == "prepare":
        result = prepare(args.source, args.dataset, args.output)
    elif args.command == "aggregate":
        result = aggregate_review(args.manifest, args.assessments, args.output)
    else:
        result = render_review(args.summary, args.manifest, args.output)
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
