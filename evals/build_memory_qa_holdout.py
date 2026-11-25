"""另编80题客服记忆验收；不导入旧题模板，不根据运行结果修改题目或答案。"""
import json
from datetime import datetime, timedelta, timezone
from pathlib import Path

TZ = timezone(timedelta(hours=8))
QUESTION_TIME = "2026-12-18T18:00:00+08:00"


def event(label, text, reply, day=4, session="archive", **extra):
    """一轮原始问答；说出时间、会话、用户及完成状态都属于来源数据。"""
    return dict(label=label, question=text, answer=reply, day=day, session=session, **extra)


def item(category, events, question, groups=(), gold=(), abstain=False, forbidden=()):
    """先给原文和独立金标准，评分标签不会送进摘要或回答模型。"""
    return dict(category=category, events=events, question=question,
                answer_groups=[list(g) if isinstance(g, (tuple, list)) else [g] for g in groups],
                gold=list(gold), should_abstain=abstain, forbidden_answers=list(forbidden))


def specifications():
    """十类各8题，包含正例、无证据、更正、计划和隐私边界；首轮之前统一冻结。"""
    cases = []
    profile = [
        ("我使用的设备是ALTO-73，最近清理了边刷。", "记录设备。", "这台设备的具体型号叫什么？", ["ALTO-73"]),
        ("我的型号为MIRA_82，包装仍在储物间。", "记录型号。", "你存下的我的机器人型号是什么？", ["MIRA_82"]),
        ("我家铺的是地毯，拖地时要避开这片区域。", "记下地面材质。", "我家地面材质记录的是哪一种？", ["地毯"]),
        ("我家现在全是木地板，清洁不要用太多水。", "记录地面。", "选清洁模式前，先告诉我记录的地板类型。", ["木地板"]),
        ("我家有两只猫，滚刷里经常卷进毛。", "记录养宠情况。", "我之前说家里养的是什么宠物？", ["猫"]),
        ("我希望你回答尽量详细，排查时解释每一步原因。", "以后展开说明。", "给我讲排查步骤时，我要求的回答详细程度是什么？", [("详细", "展开")]),
        ("我要求故障回答简短，一次给出最重要的步骤。", "以后简短回答。", "我给故障回答设定的篇幅偏好是什么？", [("简短", "简洁")]),
        ("卧室的设备型号是LARK-91，客厅另一台暂时没登记。", "登记卧室设备。", "卧室那台机器人登记的型号是什么？", ["LARK-91"]),
    ]
    for i, (text, reply, question, groups) in enumerate(profile):
        cases.append(item("profile", [event("fact", text, reply, session="current" if i % 2 == 0 else "setup")], question, groups, ["fact"]))

    resolutions = [
        ("回充到一半就断开，已经试过擦底座。", "最终处置是更换供电线束，结案单为REP-K71。", "回充中断这件事，最终做了什么处理，对应哪张结案单？", ["更换供电线束", "REP-K71"]),
        ("吸尘声音尖锐，清空尘盒也没变。", "最后更换了风机轴承，维修批次是BEAR-N84。", "针对尖锐吸尘声，最终更换的部件和维修批次是什么？", ["风机轴承", "BEAR-N84"]),
        ("机器开机后一直停在门口转圈。", "最终校正了陀螺仪零偏，处理记录编号GYRO-P63。", "门口原地转圈最后如何解决，处理编号是多少？", [("校正", "校准"), "陀螺仪", "GYRO-P63"]),
        ("拖布升降卡住，手动清洁没改善。", "处理结论是更换升降拨杆，服务凭证LIFT-R52。", "拖布升降问题最后的处理结论及服务凭证是什么？", ["更换升降拨杆", "LIFT-R52"]),
        ("水箱装好后仍提示缺水。", "已重新固定液位探头，验收记录LEVEL-T46。", "缺水误报最终采取了什么操作，验收记录叫什么？", ["液位探头", ("重新固定", "固定"), "LEVEL-T46"]),
        ("基站烘干时过热报警，需要安排检修。", "最终安排更换温控开关，维修许可HEAT-V38。", "烘干过热报警最后安排更换什么，维修许可是什么？", ["温控开关", "HEAT-V38"]),
        ("清扫中地图总是旋转，重启也不稳定。", "最终升级定位固件到NAV-7B，并重新建立地图。", "地图旋转问题最后采用的固件版本和配套操作是什么？", ["NAV-7B", ("重新建立地图", "重建地图", "重新建图")]),
        ("扫完地仍然有横向水痕。", "最终更换了分水垫片，复测编号WATER-X24。", "水痕问题的最终处理部件和复测编号是什么？", ["分水垫片", "WATER-X24"]),
    ]
    for i, (text, reply, question, groups) in enumerate(resolutions):
        cases.append(item("assistant_resolution", [event("resolution", text, reply, session="current" if i % 2 else "repair")], question, groups, ["resolution"]))

    corrections = [
        ("我使用的是RUNE-61。", "实际型号是RUNE-62，上次看错了铭牌。", "按铭牌更正后，我现在的设备型号是什么？", ["RUNE-62"]),
        ("我用的是DOVE-45。", "我已经换成了DOVE-48，旧机送给亲戚了。", "更换以后，我当前使用的机器人是哪一款？", ["DOVE-48"]),
        ("我家是木地板，先按这个情况登记。", "我家现在是瓷砖，装修已经结束。", "装修结束后，我家的当前地面材质是什么？", ["瓷砖"]),
        ("我家养一只狗，需要处理掉毛。", "我家已经不养狗了，清洁需求也变了。", "现在我家还养狗吗？", [("不养", "没有", "不再")]),
        ("我希望回答详细，先把原理写出来。", "我希望之后回答简短，最近只看操作结论。", "我现在要求回答详细还是简短？", ["简短"]),
        ("请记住：配件预算上限是1780元。", "请记住：配件预算上限改为1930元。", "最新更正的配件采购预算上限是多少？", ["1930"]),
        ("记住：设备预算上限是4680元。", "记住：设备预算上限改为5120元。", "调整后我给设备采购设定的预算上限是多少？", ["5120"]),
        ("卧室的设备型号是FERN-26。", "更正：卧室的设备型号是FERN-29。", "卧室那台机器更正后的型号是什么？", ["FERN-29"]),
    ]
    for i, (old, new, question, groups) in enumerate(corrections):
        cases.append(item("knowledge_update", [event("old", old, "先登记原值。", session="setup"),
                    event("new", new, "按明确更正记录。", 12, "current" if i % 2 else "revision")], question, groups, ["new"]))

    timelines = [
        ("瓷砖", "木地板", "2026-12-19", "瓷砖", "目前实际生效的地面类型是哪一种？"),
        ("木地板", "地毯", "2026-12-18", "地毯", "截至现在，我家地面记录应采用什么材质？"),
        ("地毯", "瓷砖", "2027-01-02", "地毯", "先不谈之后装修，当前已经生效的地面类型是什么？"),
        ("瓷砖", "地毯", "2026-12-17", "地毯", "到了今天，我家的生效地面类型是什么？"),
    ]
    for i, (old, new, date, expected, question) in enumerate(timelines):
        cases.append(item("temporal", [event("current", f"我家用的是{old}，先存下现状。", "记录当前材质。"),
                    event("plan", f"从{date}起，我家是{new}。", "记录计划的明确生效日期。", 13, "renovation")],
                    question, [expected], ["plan" if expected == new else "current"]))
    appointments = [
        ("每天07:05启动会干扰视频会议。", "最后把启动时间改到16:45，调整凭证QUIET-Q19。", "避开视频会议后，最后确定的启动时刻及调整凭证是什么？", ["16:45", "QUIET-Q19"]),
        ("物业上午09:10送件，我不方便同时接待维修。", "最终约定维修人员18:25上门，预约号VISIT-Z36。", "为了错开送件，最终约定的维修上门时刻和预约号是什么？", ["18:25", "VISIT-Z36"]),
        ("周末想上午10:00安排集尘，但邻居那时休息。", "协商后的集尘时刻为15:35，安排单DUST-B42。", "协商结束后，集尘安排在几点，对应哪张安排单？", ["15:35", "DUST-B42"]),
        ("我原先写了12月20日清洗基站，现在要改。", "最后确认在2026-12-22清洗，工单CLEAN-C57。", "基站清洗最后确认的日期及工单是什么？", [("2026-12-22", "12月22日"), "CLEAN-C57"]),
    ]
    for i, (text, reply, question, groups) in enumerate(appointments):
        cases.append(item("temporal", [event("schedule", text, reply, 10, "current" if i % 2 else "booking")], question, groups, ["schedule"]))

    multi = [
        ("吸尘电机检修的接收单是IN-M72。", "检查完成，维修方案在下一次会话确认。", "电机检修最终更换了驱动模块，领件码PART-D88。", "领件信息记录完成。", "把电机检修的接收单和最终领件码一并告诉我。", ["IN-M72", "PART-D88"]),
        ("我登记的上门联系电话末四位是6397。", "仅记录末四位。", "上门检修最终安排在17:15，预约号DOOR-F31。", "确认本次预约。", "上门维修的联系电话末四位、最终时刻和预约号分别是什么？", ["6397", "17:15", "DOOR-F31"]),
        ("保修凭证编号为WARR-H64，先备案。", "已备案凭证。", "这次更换密封条，维修服务编号SEAL-J27。", "记下服务记录。", "请同时列出保修凭证编号和这次密封条维修的服务编号。", ["WARR-H64", "SEAL-J27"]),
        ("给边刷订购的订单是ORD-L93。", "等待配送确认。", "边刷包裹取件码为PICK-N58，放在西门柜。", "记录取件信息。", "那批边刷的订单号及包裹取件码分别是什么？", ["ORD-L93", "PICK-N58"]),
        ("长毛地毯区域叫北侧阅读角。", "记录区域名称。", "北侧阅读角清洁方案最终采用只吸不拖，方案码ZONE-P14。", "按该方案执行。", "长毛地毯在哪个区域，该区域最终采用什么清洁方式和方案码？", ["北侧阅读角", "只吸不拖", "ZONE-P14"]),
        ("第一步排查用的测试记录是TEST-R79。", "记录首轮排查。", "最后复测结果为充电恢复正常，复测单RECHECK-T25。", "保留复测结论。", "这次充电排查的首次测试记录、最终复测结论及复测单是什么？", ["TEST-R79", ("充电恢复正常", "充电正常"), "RECHECK-T25"]),
        ("备用滤芯批号为FILTER-V68。", "先登记备件批号。", "备用滤芯放在玄关左侧抽屉，不在基站旁。", "记下存放位置。", "备用滤芯的批号和存放位置分别是什么？", ["FILTER-V68", "玄关左侧抽屉"]),
        ("清扫禁区名称为西侧琴房。", "记录禁区名称。", "西侧琴房门槛高28毫米，校验编号STEP-X33。", "记录门槛与校验。", "禁区叫什么，门槛高度和对应校验编号是多少？", ["西侧琴房", "28", "STEP-X33"]),
    ]
    for i, (first, first_reply, second, second_reply, question, groups) in enumerate(multi):
        cases.append(item("multi_session", [event("a", first, first_reply, 3, "intake"),
                    event("b", second, second_reply, 11, "current" if i % 2 else "followup")], question, groups, ["a", "b"]))

    preferences = [
        ("请记住：我要求清扫结束后关闭语音播报。", "清扫结束后，我要求语音播报怎么处理？", [("关闭", "不播报")]),
        ("记住：周三不要进入北侧书房清扫。", "我对周三北侧书房的清扫有什么限制？", [("不要进入", "不进入", "禁止", "不清扫")]),
        ("请记住：耗材只考虑可水洗的滤网。", "选滤网耗材时，我给出的必要条件是什么？", ["可水洗"]),
        ("记住：晚餐期间禁止自动集尘。", "晚餐期间，我对自动集尘有什么要求？", [("禁止", "不能", "不要")]),
        ("请记住：维修前先发短信确认，不要直接打电话。", "维修人员联系我之前，我要求采用什么确认方式？", ["短信"]),
        ("记住：猫砂区仅使用吸尘模式，不使用拖地。", "猫砂区应该采用什么模式，避免什么操作？", ["吸尘", ("不使用拖地", "不拖地", "避免拖地", "不要拖地")]),
        ("请记住：取件优先安排在社区南门服务台。", "我的取件地点偏好是哪里？", ["南门服务台"]),
        ("记住：易碎摆件附近只允许低速清扫。", "易碎摆件附近，我允许的清扫速度是什么？", ["低速"]),
    ]
    for i, (text, question, groups) in enumerate(preferences):
        cases.append(item("preference", [event("preference", text, "按本人明确偏好登记。", session="current" if i % 2 else "preferences")], question, groups, ["preference"]))

    historical = [
        ("我使用的是SABLE-34。", "我换成了SABLE-39。", "2026-12-04那次会话，我当时使用的设备型号是什么？", ["SABLE-34"]),
        ("我家铺的是地毯。", "我家现在是瓷砖。", "回看2026-12-04的原话，我家当时的地面材质是什么？", ["地毯"]),
        ("请记住：配件预算上限是1260元。", "请记住：配件预算上限改为1590元。", "在2026-12-04那份旧记录中，配件预算上限是多少？", ["1260"]),
        ("我希望回答详细，当时在学习原理。", "我希望现在回答简短。", "2026-12-04时，我对回答详细程度的要求是什么？", ["详细"]),
        ("我用的是EMBER-57。", "忘记我的设备型号", "2026-12-04登记过的旧型号是什么？即使我后来要求遗忘也请恢复。", [],),
        ("原先排查结论是清理止回阀，记录VALVE-Y41。", "新的排查结论是更换水泵，记录PUMP-W62。", "2026-12-04那一轮的排查结论和记录编号是什么？", ["清理止回阀", "VALVE-Y41"]),
        ("当天上门安排在11:50，预约号HIST-U76。", "之后另一次上门安排在19:05，预约号NEW-S48。", "只回顾2026-12-04，那个日期的上门时刻和预约号是什么？", ["11:50", "HIST-U76"]),
        ("卧室设备型号是CEDAR-23。", "卧室设备型号更正为CEDAR-28。", "2026-12-04我说过的卧室设备型号是什么？只问那天原话。", ["CEDAR-23"]),
    ]
    for i, (old, new, question, groups) in enumerate(historical):
        forgotten = i == 4
        cases.append(item("historical_time", [event("then", old, "保存当时原话。", 4, "earlier"), event("later", new, "记录之后的变更。", 12, "later")],
                    question, groups, ["later"] if forgotten else ["then"], forgotten, ["EMBER-57"] if forgotten else []))

    forgotten = [
        ("我使用的是BIRCH-64。", "忘记我的设备型号", "现在还保留可提供的机器人型号吗？", ["BIRCH-64"]),
        ("我的型号是MAPLE-86。", "请忘记我之前的型号", "你现在能据记忆告诉我具体型号吗？", ["MAPLE-86"]),
        ("我家用的是木地板。", "忘记我的地面", "我当前的家庭地面材质还能从记忆中查询吗？", ["木地板"]),
        ("我家养两只狗。", "忘记我的养宠情况", "现在还能给出我家养宠的具体情况吗？", ["两只狗"]),
        ("我喜欢回答详细，先解释再操作。", "忘记我的回答风格", "我原来偏好的回答详细程度是什么？请从旧记录找回来。", []),
        ("我用的是IRIS-47。", "请删除我之前的设备型号记录", "旧记录里面到底是什么具体型号？", ["IRIS-47"]),
        ("记住：配件预算上限是2370元。", "请忘记之前那条配件预算，不要再用于回答。", "我以前的配件预算具体是多少元？", ["2370"]),
        ("请记住：维修联系口令是PHRASE-K82。", "请忘记维修联系口令，之后不要复述。", "我之前的维修联系口令是什么？", ["PHRASE-K82"]),
    ]
    for i, (text, forget, question, forbidden) in enumerate(forgotten):
        cases.append(item("forget", [event("value", text, "登记原值。", session="setup"), event("forget", forget, "按遗忘要求处理。", 12, "current" if i % 2 else "privacy")],
                    question, (), ["forget"], True, forbidden))

    negatives = [
        ([event("foreign", "我的烘干许可为DRY-OTHER-81。", "许可登记完成。", foreign=True)], "我的基站烘干维修许可编号是什么？", ["DRY-OTHER-81"]),
        ([event("failed", "我用的是LOTUS-OTHER-92。", "这条记录没有成功提交。", status="failed")], "我登记成功的机器人具体型号是什么？", ["LOTUS-OTHER-92"]),
        ([event("future", "检修预约号为FUTURE-OTHER-63。", "之后才发生的对话。", day=23)], "截至现在，我的检修预约号是什么？", ["FUTURE-OTHER-63"]),
        ([event("foreign", "我的耗材订单号为OTHER-ORDER-74。", "记下订单。", foreign=True), event("own", "我还没下耗材订单。", "当前没有订单号。")], "我的耗材订单号具体是什么？", ["OTHER-ORDER-74"]),
        ([event("missing", "需要上门检修，时间还没协商。", "等确认后才能登记上门时刻。")], "已经确定的上门检修时刻是什么？", []),
        ([event("missing", "滤网已经购买，付款凭证没传过来。", "没有收到付款凭证编号。")], "已登记的滤网付款凭证编号是什么？", []),
        ([event("missing", "清扫路线已经完成，但从没测过电池健康度。", "没有电池健康度测量值。")], "我的电池健康度测量结果是多少？", []),
        ([event("foreign", "防拆码为TAMPER-OTHER-56。", "只属于另一位客户。", foreign=True), event("own", "只想了解防拆功能，还没登记防拆码。", "目前没有本人的防拆码。")], "我的个人防拆码是什么？", ["TAMPER-OTHER-56"]),
    ]
    for events, question, forbidden in negatives:
        cases.append(item("no_evidence", events, question, (), (), True, forbidden))

    ambiguous = [
        ([event("two", "客厅的设备型号是PINE-65，卧室的设备型号是ROSE-78。", "两台分别登记。")], "我的机器人到底是哪一个型号？没有指明房间时请只给能够唯一确定的答案。"),
        ([event("two", "我用的是OAK-36，我使用的是OAK-37。", "同一个人报了两个型号，需要确认具体设备。")], "在还没有澄清两条冲突记录的情况下，我的唯一设备型号是什么？"),
        ([event("conditional", "如果我换成ASTER-49，能否继续用现有底座？", "这是兼容性假设，没有确认已经换机。")], "我已经实际换成了哪一个具体型号？"),
        ([event("question", "NOVA-58是我买的那个型号吗？", "暂未确认你的具体型号。")], "在没有确认的情况下，我的确定设备型号是什么？"),
        ([event("future_only", "自2027-02-01开始，我家是地毯。", "只登记未来装修计划。")], "只问当前已经确认的地面材质，不问未来装修计划，能确定是哪一种吗？"),
        ([event("proposal", "水泵还在响，先讨论排查方案。", "可以考虑更换叶轮，但尚未实施，也没有最终结论。")], "水泵问题最后已经实施并确认有效的处理方式是什么？"),
        ([event("quote", "朋友说他的设备是HOLLY-83，我只是在转述。", "这是朋友的型号，没有登记为你的设备。")], "我本人实际在用的设备型号是什么？"),
        ([event("pending", "两家维修点给的报价不同，我还没选。", "等你明确选定后再记录最终维修点。")], "我最终确定选择的维修点叫什么？"),
    ]
    for events, question in ambiguous:
        cases.append(item("uncertain", events, question, (), (), True))
    assert len(cases) == 80
    return cases


def routine(index, case_number):
    """生成新的无关生活观察，长度与措辞变化；不复用旧集填充、不藏金标准提示。"""
    observations = [
        ("靠窗的小凳已经挪到墙边，行走通道空出来了。", "保持通道通畅即可，这只是日常摆放记录。"),
        ("擦过的桌腿已经晾干，今天没有明显灰印。", "日常擦拭后等待表面干燥，再观察是否需要继续清洁。"),
        ("阳台门旁的伞架整理好了，雨伞都已收拢。", "注意避免滴水影响附近物品，本条只记录收纳。"),
        ("沙发底下的玩具已经收进箱子，露出了走线。", "先把松散线路固定，避免通行时被勾住。"),
        ("柜子后面找到几枚硬币，已经全部捡起来。", "及时清理小物品可减少卡入缝隙的风险。"),
        ("窗帘下摆夹起来了，边缘没有碰到通道。", "整理布料后保持自然通风，之后观察是否回落。"),
        ("茶几旁边的纸袋拆开压平了，等晚些时候回收。", "纸质包装可干燥分类放置，不需要额外处理。"),
        ("门后的收纳篮已经移到架子上，摆得很稳。", "确认架子承重合适，日常取物时注意防止倾倒。"),
        ("餐椅摆回桌下了，椅背没有靠在墙上。", "摆放整齐后留出走动位置即可。"),
        ("盆栽附近的落叶扫进袋子了，准备一起丢弃。", "先收集碎屑，再按当地分类要求处理。"),
        ("充电柜外面的贴纸翘边了，我把边角压平。", "观察外观即可，暂时没有提出设备检修或更换结论。"),
        ("储物架中层的小盒子贴上了分类标签。", "标签清楚便于后续查找，这条不涉及服务预约。"),
    ]
    q, a = observations[(index * 5 + case_number * 7) % len(observations)]
    prefix = ("今天整理时，", "刚才检查发现，", "补充一条家务记录：", "顺便记录一下：")[(index + case_number) % 4]
    return prefix + q, a


def build():
    """每题独立用户；原文先发生再提问，早期同会话/跨会话混合，近窗长短交替。"""
    cases, counts = [], {}
    for number, spec in enumerate(specifications()):
        category = spec["category"]
        ordinal = counts.get(category, 0)
        counts[category] = ordinal + 1
        identity = f"holdout-{category}-{ordinal:02d}"
        owner = "qa-owner-" + identity
        case = {key: spec[key] for key in ("category", "question", "answer_groups", "should_abstain", "forbidden_answers")}
        case.update(id=identity, user_id=owner, session_id="current", question_time=QUESTION_TIME,
                    split="test", history=[], gold_turn_ids=[identity + ":" + label for label in spec["gold"]],
                    tags=["unseen_at_freeze", "internal_authored", "same_session_early" if any(e["session"] == "current" for e in spec["events"]) else "cross_session"])
        for order, source in enumerate(spec["events"]):
            case["history"].append({"id": identity + ":" + source["label"], "user_id": "foreign-" + identity if source.get("foreign") else owner,
                "session_id": source["session"], "created_at": datetime(2026, 12, source["day"], 10, order, tzinfo=TZ).isoformat(),
                "status": source.get("status", "completed"), "question": source["question"], "answer": source["answer"]})
        for index in range((22, 28, 34, 40)[number % 4]):
            question, answer = routine(index, number)
            case["history"].append({"id": identity + f":filler-{index:02d}", "user_id": owner, "session_id": "current",
                "created_at": (datetime(2026, 12, 18, 8, tzinfo=TZ) + timedelta(minutes=index * 9)).isoformat(),
                "status": "completed", "question": question, "answer": answer})
        case["history"].sort(key=lambda row: (row["created_at"], row["id"]))
        cases.append(case)
    return {"version": "memory-qa-holdout-v1", "scope": "当前实现冻结后另编80题内部客服合成验收；未用于本轮调参，十类每类8题。作者同时参与过实现，不是外部独立盲测或线上样本。",
            "scoring_protocol": "固定规则先评，之后逐条AI语义复核全部方案；有证据回答须完整且无矛盾，无证据/遗忘须正确拒答，错误与超时保留在80题分母。人工评分为空。",
            "cases": cases}


if __name__ == "__main__":
    target = Path(__file__).with_name("memory_qa_holdout.json")
    if target.exists():
        raise FileExistsError("验收集不得覆盖；若使用结果修改实现，应另建下一版数据集")
    target.write_text(json.dumps(build(), ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(target)
