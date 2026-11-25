"""抽取与摘要的确定性指标：严格标签匹配不等于完整语义判断。"""
import re

from evals.history_metrics import contains_alias, normalize


def canonical_value(value):
    """只忽略句末标点和排版，不把不同事实内容自动视作同义。"""
    return normalize(value).strip("。；;，,！! ")


def candidate_dict(candidate):
    """将生产MemoryCandidate转为评测记录，笔记哈希归为note类别但不改其值。"""
    return {"key": "note" if candidate.key.startswith("note:") else candidate.key,
            "value": candidate.value, "entity": candidate.entity, "action": candidate.action,
            "evidence": candidate.evidence, "valid_at": candidate.valid_at}


def maximum_matches(predictions, expected):
    """一对一最大匹配，重复候选不能重复计TP；字段、对象、动作和有效时间同时正确才命中。"""
    def compatible(prediction, gold):
        """匹配预先冻结的可接受逐字值，严禁看完模型答案再扩充别名。"""
        return (all(prediction.get(key, "") == gold.get(key, "") for key in ("key", "entity", "action", "valid_at"))
                and canonical_value(prediction["value"]) in {canonical_value(value) for value in gold["values"]})

    edges = [[index for index, gold in enumerate(expected) if compatible(prediction, gold)] for prediction in predictions]
    assigned = {}

    def assign(prediction_index, visited):
        """增广路径允许重新分配别名匹配，避免贪心顺序少算TP。"""
        for gold_index in edges[prediction_index]:
            if gold_index not in visited:
                visited.add(gold_index)
                if gold_index not in assigned or assign(assigned[gold_index], visited):
                    assigned[gold_index] = prediction_index
                    return True
        return False

    for index in range(len(predictions)):
        assign(index, set())
    return sorted((prediction, gold) for gold, prediction in assigned.items())


def extraction_score(case, predictions):
    """严格候选级TP/FP/FN，原文依据与缺失证据另记；空输出不能靠负例冒充高召回。"""
    matched = maximum_matches(predictions, case["expected"])
    tp = len(matched)
    unsupported = [index for index, row in enumerate(predictions) if row["action"] != "delete" and row["value"] not in case["text"]]
    return {"tp": tp, "fp": len(predictions) - tp, "fn": len(case["expected"]) - tp,
            "gold_count": len(case["expected"]), "predicted_count": len(predictions), "matches": [list(pair) for pair in matched],
            "case_exact": tp == len(predictions) == len(case["expected"]),
            "unsupported_values": unsupported,
            "missing_explicit_evidence": sum(not row.get("evidence") for row in predictions if row["action"] != "delete")}


def extraction_aggregate(rows):
    """候选微平均与案例通过率分开，纯负例写入率单列；接口失败不会从案例分母删除。"""
    tp, fp, fn = (sum(row["score"][key] for row in rows) for key in ("tp", "fp", "fn"))
    negatives = [row for row in rows if not row["score"]["gold_count"]]
    positives = [row for row in rows if row["score"]["gold_count"]]
    return {"cases": len(rows), "tp": tp, "fp": fp, "fn": fn,
            "precision": tp / (tp + fp) if tp + fp else None,
            "recall": tp / (tp + fn) if tp + fn else None,
            "f1": 2 * tp / (2 * tp + fp + fn) if 2 * tp + fp + fn else None,
            "case_exact": sum(row["score"]["case_exact"] and not row.get("error") for row in rows),
            "positive_cases": len(positives), "negative_cases": len(negatives),
            "negative_false_write_cases": sum(row["score"]["predicted_count"] > 0 for row in negatives),
            "unsupported_value_count": sum(len(row["score"]["unsupported_values"]) for row in rows),
            "cases_with_failed_model_calls": sum(any("error" in event for event in row["calls"]) for row in rows),
            "error_cases": sum(bool(row.get("error")) for row in rows)}


def fact_present(text, groups):
    """必要信息必须在同一短句/相邻片段内出现，不能把相距很远的词拼成一个事实。"""
    sentences = re.split(r"[。\n；;]", text)
    fragments = [sentence[index:index + 180] for sentence in sentences for index in range(0, max(1, len(sentence)), 90)]
    return any(all(any(contains_alias(fragment, alias) for alias in group) for group in groups) for fragment in fragments)


def summary_score(case, summary):
    """统计关键事实保留、已知错误声明和无原文依据标识符；不宣称检测所有语义幻觉。"""
    retained = {fact["id"]: fact_present(summary, fact["groups"]) for fact in case["facts"]}
    wrong = []
    for claim in case["known_wrong_claims"]:
        for match in re.finditer(claim["pattern"], summary):
            prefix = re.split(r"[。\n；;，,]", summary[:match.start()])[-1][-8:]
            if not re.search(r"不|未|否认|不能|不要", prefix):
                wrong.append(claim["label"])
    source = case["old_summary"] + "\n" + "\n".join(message["content"] for message in case["messages"])
    identifiers = re.findall(r"(?<![A-Za-z0-9])[A-Za-z]{1,8}-?\d{3,}(?![A-Za-z0-9])", summary)
    unsupported = [identifier for identifier in dict.fromkeys(identifiers) if not contains_alias(source, identifier)]
    return {"retained": sum(retained.values()), "required": len(retained), "fact_results": retained,
            "retention_recall": sum(retained.values()) / len(retained) if retained else None,
            "all_key_facts_retained": all(retained.values()), "known_wrong_claims": wrong,
            "unsupported_identifiers": unsupported}


def summary_aggregate(rows):
    """微平均关键事实保留率，失败按零保留计；输出压缩率仅计算已成功输出的案例。"""
    retained = sum(row["score"]["retained"] for row in rows)
    required = sum(row["score"]["required"] for row in rows)
    completed = [row for row in rows if not row.get("error")]
    input_bytes = sum(row["source_bytes"] for row in completed)
    return {"cases": len(rows), "retained_facts": retained, "required_facts": required,
            "fact_retention_recall": retained / required if required else None,
            "all_facts_retained_cases": sum(row["score"]["all_key_facts_retained"] for row in rows),
            "known_wrong_claim_cases": sum(bool(row["score"]["known_wrong_claims"]) for row in rows),
            "unsupported_identifier_cases": sum(bool(row["score"]["unsupported_identifiers"]) for row in rows),
            "failed_cases": len(rows) - len(completed),
            "multi_batch_cases": sum(row.get("batches", 0) > 1 for row in rows),
            "output_to_source_byte_ratio": sum(row["summary_bytes"] for row in completed) / input_bytes if input_bytes else None}
