"""新增记忆修复验收集，题目/编号独立编写；不依据模型输出筛选或修改标签。"""
import json
import sys
from datetime import datetime, timedelta, timezone
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from evals.build_history_cases import FILLERS

TZ = timezone(timedelta(hours=8))


def build_case(category, number):
    """构造预算、实际生效边界、新时间与遗忘案例，每题都有原文金标准。"""
    identity = f"refinement-{category}-{number:02d}"
    case = {"id": identity, "split": "test", "category": category, "tags": [], "user_id": "owner-" + identity,
            "session_id": "current", "question_time": datetime(2026, 11, 24, 18, tzinfo=TZ).isoformat(),
            "history": [], "question": "", "gold_turn_ids": [], "answer_groups": [], "forbidden_answers": [], "should_abstain": False}

    def add(label, question, answer, day, gold=False, session="old"):
        """将正确来源作为稳定轮次ID，未来计划的说出时间与生效日期分开。"""
        row_id = identity + ":" + label
        case["history"].append({"id": row_id, "user_id": case["user_id"], "session_id": session,
             "created_at": datetime(2026, 11, day, 10, tzinfo=TZ).isoformat(), "status": "completed", "question": question, "answer": answer})
        if gold:
            case["gold_turn_ids"].append(row_id)

    if category == "budget_revision":
        value, changed = 3100 + number * 50, 3300 + number * 50
        add("old", f"记住：配件预算上限是{value}元。", "记录预算。", 3)
        add("new", f"记住：配件预算上限改为{changed}元。", "按更正后的上限。", 22, True)
        case.update(question="更正以后，我现在的配件预算上限是多少？只回答当前金额。", answer_groups=[[str(changed)]])
    elif category == "effective_date":
        # 计划均在提问前说出：0~2未到期，3~5到期。不是把未来消息过滤掉的伪时间验证。
        effective_day = 25 if number < 3 else 24
        add("current", "我家是瓷砖。", "已记录当前地面。", 3, number < 3)
        add("planned", f"从2026-11-{effective_day}起，我家是木地板。", "记录明确生效日期。", 22, number >= 3)
        case.update(question="截至提问时，我家已经生效的地面类型是什么？只回答当前类型，不提供未来计划。",
                    answer_groups=[["瓷砖" if number < 3 else "木地板"]], tags=["stored_future_plan" if number < 3 else "due_plan"])
    elif category == "changed_schedule":
        hour, minute, code = 13 + number % 3, 10 + number * 5, f"AP-{5100 + number}"
        add("schedule", "机器人早晨六点噪音太大吵醒家人。", f"最终改为{hour}:{minute:02d}运行，安排编号{code}。", 3, True, "current")
        case.update(question="为了避开早晨噪音，最终调整后的运行时刻及安排编号是什么？",
                    answer_groups=[[f"{hour}:{minute:02d}", f"{hour}点{minute}分"], [code]])
    else:
        model = f"Z{6100 + number}"
        add("model", f"我用的是{model}。", "已登记型号。", 3)
        add("forget", "忘记我的设备型号", "不再使用旧型号记录。", 22, True)
        case.update(question="我之前要求删除型号记录；现在还有可提供的型号吗？不要重述已删除的具体值。",
                    should_abstain=True, forbidden_answers=[model], tags=["forget"])
    for index in range(30):
        question, answer = FILLERS[index % len(FILLERS)]
        case["history"].append({"id": identity + f":filler-{index:02d}", "user_id": case["user_id"], "session_id": "current",
            "created_at": (datetime(2026, 11, 24, 8, tzinfo=TZ) + timedelta(minutes=index * 10)).isoformat(),
            "question": question, "answer": answer, "status": "completed"})
    case["history"].sort(key=lambda row: (row["created_at"], row["id"]))
    return case


def build():
    """四类各6题，共24个内部合成新验收案例；旧失败案例只用于回归。"""
    return {"version": "memory-refinement-v1", "scope": "修复后新增内部合成验收集，模板相似性存在，未经人工二次审核，不是外部盲测。",
            "cases": [build_case(category, number) for category in ("budget_revision", "effective_date", "changed_schedule", "forget") for number in range(6)]}


if __name__ == "__main__":
    path = Path(__file__).with_name("refinement_cases.json")
    path.write_text(json.dumps(build(), ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(path)
