"""生成首版合成历史验收集；标签只取自可见原文，不读取检索器或模型输出。"""
import json
from datetime import datetime, timedelta, timezone
from pathlib import Path


TZ = timezone(timedelta(hours=8))
CATEGORIES = ("exact", "assistant_resolution", "paraphrase", "multi_session", "temporal", "knowledge_update", "no_answer", "recent")
FILLERS = (
    ("今天擦拭了外壳，表面没有发现划痕。", "继续用柔软的干布擦拭即可，不要让水进入机身接口；这条仅作为日常外观检查记录。"),
    ("收纳时把附件重新放回盒子了。", "附件按原有隔层放置，避免运输时互相碰撞；本轮没有安排维修，也没有新增订单或预约。"),
    ("检查了窗边的摆放位置，周围是空的。", "保持通道通畅并避免暴晒，暂时无需更换设备或购买配件；这只是环境检查的后续反馈。"),
    ("我把包装纸和快递箱单独整理好了。", "包装材料可以分类收纳，保留说明书即可；这一轮不涉及产品型号、验证码或保修合同。"),
    ("这次查看了连接线，表面都完好。", "暂时保持现有连接方式，不要反复插拔；目前只记录观察结果，还没有形成新的故障处理结论。"),
)


def timestamp(day, hour=10, minute=0):
    """固定UTC+8时间，避免生成的数据随运行日期改变。"""
    return datetime(2026, 11, day, hour, minute, tzinfo=TZ).isoformat()


def make_case(split, category, number):
    """建立独立用户/会话的一个问题，包含无关历史、正确证据和必要的干扰记录。"""
    identity = f"{split}-{category}-{number:02d}"
    place = (["校准仓", "校准工位"][number % 2] if split == "calibration" else
             ["储物间", "门厅", "地下室", "客房", "餐厅", "走廊"][number])
    base = (1000 if split == "calibration" else 2000) + CATEGORIES.index(category) * 100 + number
    case = {"id": identity, "split": split, "category": category, "tags": [],
            "user_id": "owner-" + identity, "session_id": "current", "question_time": timestamp(24, 18),
            "question": "", "history": [], "gold_turn_ids": [], "answer_groups": [],
            "forbidden_answers": [], "should_abstain": False}

    def add(label, question, answer, day=3, session="old", user=None, hour=10, status="completed", gold=False):
        """只维护稳定来源标签；SQL运行时ID由评测器单独映射，不进入参考答案。"""
        turn_id = identity + ":" + label
        case["history"].append({"id": turn_id, "user_id": user or case["user_id"], "session_id": session,
                                "created_at": timestamp(day, hour), "question": question, "answer": answer, "status": status})
        if gold:
            case["gold_turn_ids"].append(turn_id)
        return turn_id

    if category == "exact":
        code = f"LC-{base}"
        add("batch", f"请登记{place}设备的配件批次。", f"当时登记的配件批次是{code}。", session="current" if number % 2 == 0 else "old", gold=True)
        case.update(question=f"之前{place}设备登记的配件批次号是多少？", answer_groups=[[code]])
    elif category == "assistant_resolution":
        code = f"WX-{base}"
        action = ["送修电源板", "更换驱动轮", "重新固定传感器", "清理充电触点", "更换密封圈", "校准定位模块"][number]
        add("repair", f"{place}设备反复停机，你最终给出的处理结论是什么？", f"最后确定的办法是{action}，维修授权码为{code}。", session="current" if number < 2 else "old", gold=True)
        case.update(question=f"之前{place}那次停机最后决定怎么处理，维修授权码是什么？", answer_groups=[[action], [code]])
    elif category == "paraphrase":
        stories = [
            ("那台机器凌晨轰隆作响，把宝宝惊醒了。", f"我们约定改到上午十点四十五分启动，安排编号JH-{base}。", "以前为了避免打扰家人休息，运行时间定在几点，安排编号是多少？", [["10:45", "十点四十五分", "10点45分"], [f"JH-{base}"]]),
            ("我不想再买价格高的那类款式。", f"当时确定的采购金额天花板为{base}元。", "之前那次选购，我最多愿意花多少钱？", [[str(base)]]),
            ("家里金毛掉得厉害，滚刷总是被缠住。", f"当时选定使用防缠绕套件，套件代号FC-{base}。", "之前针对宠物毛发卡住刷子的情况，选了什么附件、代号是多少？", [["防缠绕套件", "防缠绕组件"], [f"FC-{base}"]]),
            ("搬到新住处后，手机始终找不到机器。", f"当时约定重置配网，工单编号NET-{base}。", "以前迁居后那次联网失败，最终做什么，工单编号是多少？", [["重置配网", "重新配网"], [f"NET-{base}"]]),
            ("橱柜下面总有一块位置擦不到。", f"当时采用安装窄边刷的办法，附件批号ZB-{base}。", "之前家具底部出现清洁死角，加装了什么配件、批号是多少？", [["窄边刷"], [f"ZB-{base}"]]),
            ("扫完屋子仍然能闻到一股怪味。", f"当时决定更换除味滤芯，备件编号CW-{base}。", "以前室内异味残留，最后换了什么，备件编号是多少？", [["除味滤芯"], [f"CW-{base}"]]),
        ]
        question, answer, query, groups = stories[number]
        add("paraphrase", question, answer, session="current" if number < 2 else "old", gold=True)
        case.update(question=query, answer_groups=groups)
    elif category == "multi_session":
        code, day = f"GD-{base}", 25 + number % 4
        add("ticket", f"{place}那台设备的维修登记做完了。", f"对应维修登记号为{code}。", session="ticket", gold=True)
        add("appointment", f"请继续安排登记号{code}的上门。", f"已约定2026年11月{day}日14:20上门。", day=6, session="appointment", gold=True)
        case.update(question=f"之前{place}设备的维修登记号，以及对应上门日期和时间分别是什么？",
                    answer_groups=[[code], [f"11月{day}日", f"2026-11-{day}"], ["14:20", "14点20分", "下午2点20分"]])
    elif category == "temporal":
        old, correct, future = f"OLD-{base}", f"LAST-{base}", f"FUT-{base}"
        add("earlier", f"{place}的设备报警了。", f"当时决定断电静置，处理码{old}。", day=22)
        add("yesterday", f"{place}的设备再次报警。", f"最后决定更换驱动轮，处理码{correct}。", day=23, gold=True)
        add("future", f"{place}报警后新安排。", f"未来决定更换主板，处理码{future}。", day=25)
        case.update(question=f"昨天针对{place}设备报警的最终处理是什么、处理码是多少？",
                    answer_groups=[["更换驱动轮"], [correct]], forbidden_answers=[future], tags=["future_exclusion"])
    elif category == "knowledge_update":
        old, new = f"X{base}", f"V{base}"
        add("model-old", f"我用的是{old}。", "已记录当时的设备型号。", gold=number in (1, 2))
        if number == 0:
            add("model-new", f"实际型号是{new}。", "以本次更正为准。", day=22, gold=True)
            case.update(question="我现在使用的设备型号是什么？", answer_groups=[[new]])
        elif number == 1:
            add("future-model", f"实际型号是{new}。", "此更正发生在提问之后。", day=25)
            case.update(question="截至现在，我已经登记的设备型号是什么？", answer_groups=[[old]], forbidden_answers=[new], tags=["future_exclusion"])
        elif number == 2:
            add("model-new", f"实际型号是{new}。", "已更新当前型号。", day=22)
            case.update(question="2026-11-03那天我说使用的设备型号是什么？", answer_groups=[[old]], tags=["historical_value"])
        elif number == 3:
            add("style-old", "我喜欢回答详细。", "已记录旧偏好。", day=4)
            add("style-new", "我希望回答简短。", "以后按新偏好回答。", day=22, gold=True)
            case.update(question="我目前要求回答采用什么风格？", answer_groups=[["简短", "简洁"]])
        elif number == 4:
            add("budget-old", f"记住：配件预算上限是{base}元。", "记录预算。", day=4)
            add("budget-new", f"记住：配件预算上限改为{base + 100}元。", "已记录更正后的预算。", day=22, gold=True)
            case.update(question="之前更正后，我现在的配件预算上限是多少？", answer_groups=[[str(base + 100)]])
        else:
            add("forget", "忘记我的设备型号", "不再把旧型号作为当前记忆使用。", day=22, gold=True)
            case.update(question="我现在可用的设备型号记录是什么？如果已要求遗忘，请不要恢复旧值。",
                        should_abstain=True, forbidden_answers=[old], tags=["forget"])
    elif category == "no_answer":
        secret = f"PRIVATE-{base}"
        if number < 3:
            add("foreign", f"请登记{place}的授权口令。", f"口令是{secret}。", user="foreign-" + identity)
            case.update(question=f"我之前为{place}登记的授权口令是什么？", tags=["user_isolation"])
        elif number == 3:
            add("unrelated", "记录一下机器颜色是白色。", "已记录外观。")
            case.update(question="我的保修合同编号是什么？")
        elif number == 4:
            add("future-only", "先把应急解锁码存着。", f"应急解锁码{secret}。", day=25)
            case.update(question="截至现在，我已登记的应急解锁码是什么？", tags=["future_exclusion"])
        else:
            add("failed", f"我提出过授权口令{secret}，但请求中断。", "", status="failed")
            case.update(question="我之前成功登记并得到确认的授权口令是什么？", tags=["failed_turn"])
        case.update(should_abstain=True, forbidden_answers=[secret])
    elif category == "recent":
        code = f"RJ-{base}"
        add("recent", f"{place}这次取件确认完成了吗？", f"已经完成，取件验证码为{code}。", day=24, hour=16, session="current", gold=True)
        case.update(question=f"刚才{place}约定的取件验证码是什么？", answer_groups=[[code]])

    # 足够长的当前会话让“仅近期窗口”确实不能看到更早会话，不加入答案相关事实。
    for index in range(30):
        question, answer = FILLERS[index % len(FILLERS)]
        moment = datetime(2026, 11, 24, 8, tzinfo=TZ) + timedelta(minutes=index * 10)
        case["history"].append({"id": identity + f":filler-{index:02d}", "user_id": case["user_id"],
                                "session_id": "current", "created_at": moment.isoformat(),
                                "question": question, "answer": answer, "status": "completed"})
    case["history"].sort(key=lambda turn: (turn["created_at"], turn["id"]))
    return case


def build():
    """按会话独立生成12例校准和48例验收；不根据检索成绩筛选或删除案例。"""
    calibration = [make_case("calibration", category, number)
                   for category in CATEGORIES[:6] for number in range(2)]
    heldout = [make_case("test", category, number) for category in CATEGORIES for number in range(6)]
    return {"version": "history-v1", "created_on": "2026-11-24", "seed": "deterministic-no-random",
            "scope": "新编写的合成标注集，独立于既有30例；同类题存在模板相似性，未经人工二次审核，不是外部盲测集。",
            "protocol": "先冻结SHA256，再运行校准与验收；未来、其他用户、失败轮次不得进入回答上下文。",
            "cases": calibration + heldout}


if __name__ == "__main__":
    destination = Path(__file__).with_name("history_cases.json")
    destination.write_text(json.dumps(build(), ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(destination)
