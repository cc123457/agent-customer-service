"""历史评测的标签校验与确定性评分；不导入模型，不参与生产检索。"""
import math
import re
import unicodedata
from datetime import datetime


def visible_history(case, *, own_only=False, completed_only=False):
    """模拟提问时已发生的数据；未来轮次不入库，外人/失败轮次保留用于SQL隔离检查。"""
    now = datetime.fromisoformat(case["question_time"])
    return [row for row in case["history"]
            if datetime.fromisoformat(row["created_at"]) <= now
            and (not own_only or row["user_id"] == case["user_id"])
            and (not completed_only or row["status"] == "completed")]


def validate_dataset(dataset):
    """在调用任何模型前校验来源、正负标签和拆分独立性，错误标签不能静默进入分母。"""
    cases = dataset["cases"]
    assert len({case["id"] for case in cases}) == len(cases), "案例ID重复"
    users = {}
    for case in cases:
        assert case["question"].strip() and case["split"] in ("calibration", "test")
        assert case["user_id"] not in users, "案例必须使用独立用户"
        users[case["user_id"]] = case["id"]
        assert len({row["id"] for row in case["history"]}) == len(case["history"])
        visible = {row["id"]: row for row in visible_history(case, own_only=True, completed_only=True)}
        assert set(case["gold_turn_ids"]).issubset(visible), "金标准引用了不可见来源"
        assert case["should_abstain"] or (case["answer_groups"] and case["gold_turn_ids"])
        sources = "\n".join(visible[key]["question"] + visible[key]["answer"] for key in case["gold_turn_ids"])
        for group in case["answer_groups"]:
            assert group and any(alias in sources for alias in group), "答案没有原文依据"
    return cases


def retrieval_score(case, returned, k=3):
    """Hit按问题计、Recall按每题证据覆盖计、MRR按首个正确证据排名计；负例不混入正例召回。"""
    ranked = list(dict.fromkeys(returned))[:k]
    gold = set(case["gold_turn_ids"])
    hits = gold.intersection(ranked)
    first = next((index for index, key in enumerate(ranked, 1) if key in gold), None)
    allowed = {row["id"] for row in visible_history(case, own_only=True, completed_only=True)}
    return {"gold_count": len(gold), "hit": bool(hits), "recall": len(hits) / len(gold) if gold else None,
            "reciprocal_rank": 1 / first if first else 0, "all_evidence": bool(gold) and hits == gold,
            "returned_on_negative": not gold and bool(ranked), "invalid_sources": len(set(ranked) - allowed),
            "returned_count": len(ranked), "relevant_returned": len(hits),
            "precision_at_k": len(hits) / k if gold else None,
            "precision_returned": len(hits) / len(ranked) if ranked and gold else (0 if gold else None)}


def abstention_metrics(cases, records):
    """按严格布尔拒答标记统计混淆矩阵；自然语言事实错误/泄露需另评，接口失败单列。"""
    answerable = [case for case in cases if not case["should_abstain"]]
    unanswerable = [case for case in cases if case["should_abstain"]]
    tp = fp = fn = tn = invalid = invalid_unanswerable = 0
    for case, record in zip(cases, records, strict=True):
        if not record.get("structure_valid") or type(record.get("abstain")) is not bool:
            invalid += 1
            invalid_unanswerable += case["should_abstain"]
            continue
        if case["should_abstain"]:
            tp += record["abstain"]
            fn += not record["abstain"]
        else:
            fp += record["abstain"]
            tn += not record["abstain"]
    precision = tp / (tp + fp) if tp + fp else None
    recall = tp / len(unanswerable) if unanswerable else None
    return {"should_abstain_cases": len(unanswerable), "answerable_cases": len(answerable),
            "correct_abstentions": tp, "incorrect_answer_flags": fn, "over_refusals": fp,
            "correct_answer_flags": tn, "invalid_or_missing": invalid,
            "invalid_unanswerable": invalid_unanswerable,
            "abstention_precision": precision, "abstention_recall": recall,
            "abstention_f1": 2 * tp / (2 * tp + fp + fn + invalid_unanswerable) if 2 * tp + fp + fn + invalid_unanswerable else None,
            "incorrect_answer_flag_rate": fn / len(unanswerable) if unanswerable else None,
            "over_refusal_rate": fp / len(answerable) if answerable else None}


def normalize(text):
    """统一全半角、空白和大小写，保留数字/型号边界，避免X20误命中X200。"""
    return re.sub(r"\s+", "", unicodedata.normalize("NFKC", text)).casefold()


def contains_alias(text, alias):
    """允许中文词包含匹配，数字或英文端点要求不能继续接同类字母数字。"""
    value = normalize(alias)
    left = r"(?<![a-z0-9])" if value and value[0].isascii() and value[0].isalnum() else ""
    right = r"(?![a-z0-9])" if value and value[-1].isascii() and value[-1].isalnum() else ""
    return bool(value and re.search(left + re.escape(value) + right, normalize(text)))


def answer_score(case, answer, abstain):
    """规则检查短答案、全部必需信息和拒答；human_correct另留人工复核，不能称人工准确率。"""
    matched = [any(contains_alias(answer, alias) for alias in group) for group in case["answer_groups"]]
    forbidden = [alias for alias in case["forbidden_answers"] if contains_alias(answer, alias)]
    passed = abstain == case["should_abstain"] and all(matched) and not forbidden
    return {"passed": passed, "matched_groups": sum(matched), "required_groups": len(matched),
            "forbidden_matches": forbidden, "human_correct": None}


def percentile(values, fraction):
    """对已有有效测量用最近秩统计分位数，空集合返回None而非虚假的零延迟。"""
    finite = sorted(value for value in values if value is not None and math.isfinite(value))
    return finite[max(0, math.ceil(fraction * len(finite)) - 1)] if finite else None
