"""路由/遗忘升级后新编40题；首次模型运行之前冻结，不覆盖旧80题。"""
import json
from datetime import datetime, timedelta, timezone
from pathlib import Path

TZ = timezone(timedelta(hours=8))


def build():
    """十类各四题，真值来自先写好的原文；用户、编号与旧集独立。"""
    specs = []
    def add(category, history, question, groups=(), gold=(), abstain=False, forbidden=()):
        """每条history为(标签、用户原话、助手原话、会话、日期、可选状态)。"""
        specs.append(dict(category=category, history=history, question=question, answer_groups=[[v] for v in groups],
                          gold=gold, should_abstain=abstain, forbidden_answers=list(forbidden)))
    for n in range(4):
        code=f"CIRRUS-{410+n}"
        add("profile", [("fact",f"我用的是{code}。","记录型号。","setup",2)], "我的设备型号是什么？", [code], ["fact"])
        receipt, pickup=f"RCP-{510+n}",f"TAKE-{610+n}"
        add("imperative", [("repair","电机检修已经结束。",f"检修接收单为{receipt}，最终领件码是{pickup}。","repair",3)],
            "请把电机检修接收单和领件码一并列给我。", [receipt,pickup], ["repair"])
        old,new=f"CEDAR-{710+n}",f"CEDAR-{810+n}"
        add("correction",[("old",f"我用的是{old}。","先登记。","setup",2),
                           ("new",f"实际型号是{new}，之前看错了。","更正记录。","current",12)],
            "更正后，我现在的型号是什么？",[new],["new"],forbidden=[old])
        secret=f"HIDDEN-{910+n}"
        deletion=("请忘记我之前的型号。","请删除我之前的设备型号记录。","忘记我的型号。","忘记我的设备型号。")[n]
        add("forget",[("old",f"我用的是{secret}。","已记录。","setup",2),
                       ("forget",deletion,"已收到停止使用的要求。","current",12)],
            "2026-12-02那天我使用的设备型号是什么？",[],["forget"],True,[secret])
        cutoff="2027-01-05" if n%2==0 else "2026-12-14"
        expected="瓷砖" if n%2==0 else "木地板"
        add("effective_date",[("old","我家是瓷砖。","记录现状。","setup",2),
            ("plan",f"从{cutoff}起，我家是木地板。","记录生效日期。","current",12)],
            "现在已经生效的地面材质是什么？",[expected],["old" if n%2==0 else "plan"])
        code=f"FOREIGN-{1110+n}"
        add("no_evidence",[("irrelevant","清洁刷已经收回抽屉。","保持干燥。","current",3),
            ("foreign","检修凭证是什么？",f"凭证是{code}。","foreign",4)],
            "我上次电机检修的凭证编号是多少？",[],[],True,[code])
        a,b=f"LIVING-{1210+n}",f"BEDROOM-{1310+n}"
        add("ambiguity",[("a",f"客厅设备型号是{a}。","登记客厅。","setup",2),
                         ("b",f"卧室设备型号是{b}。","登记卧室。","setup",3)],
            "我这台机器的型号是什么？",[],["a","b"],True,[a,b])
        first,second=f"WARRANTY-{1410+n}",f"SERVICE-{1510+n}"
        add("multi_session",[("warranty","保修凭证找到了。",f"凭证编号为{first}。","warranty",2),
                              ("service","密封条维修结案。",f"服务编号为{second}。","service",4)],
            "请同时列出保修凭证编号和密封条维修服务编号。",[first,second],["warranty","service"])
        style="简短" if n%2==0 else "详细"
        add("short_forget",[("preference",f"我希望回答{style}。","按此风格。","current",18),
            ("forget","忘记我的回答风格。","已收到撤回要求。","current",19)],
            "我要求的回答详细程度是什么？",[],["forget"],True,[style])
        budget=str(3200+n*110)
        add("note_forget",[("note",f"记住：配件预算上限是{budget}元。","记录预算。","setup",2),
            ("forget","忘记我的配件预算。","不再使用旧预算。","current",12)],
            "原来配件预算上限有多少元？",[],["forget"],True,[budget])
    cases=[]
    for number,spec in enumerate(specs):
        identity=f"routing-v2-{spec['category']}-{number//10:02d}"
        owner="routing-owner-"+identity
        events=[]
        for order,row in enumerate(spec["history"]):
            label,q,a,session,day=row
            events.append(dict(id=identity+":"+label,user_id="foreign-"+owner if label=="foreign" else owner,
                session_id=session,created_at=datetime(2026,12,day,9,order,tzinfo=TZ).isoformat(),
                status="completed",question=q,answer=a))
        # 长历史题把有效事实移出近期窗口；短期遗忘题刻意不填充，验证近期路径。
        if spec["category"]!="short_forget":
            for i in range(22+(number%3)*2):
                events.append(dict(id=identity+f":routine-{i}",user_id=owner,session_id="current",
                    created_at=(datetime(2026,12,20,8,tzinfo=TZ)+timedelta(minutes=i)).isoformat(),
                    status="completed",question=f"今天第{i+1}次整理了清洁柜，抹布已晾干。",answer="请保持收纳位置通风。"))
        cases.append(dict(id=identity,user_id=owner,session_id="current",question_time="2026-12-20T18:00:00+08:00",
            split="test",category=spec["category"],question=spec["question"],history=sorted(events,key=lambda e:e["created_at"]),
            answer_groups=spec["answer_groups"],gold_turn_ids=[identity+":"+v for v in spec["gold"]],
            should_abstain=spec["should_abstain"],forbidden_answers=spec["forbidden_answers"],tags=["new_at_freeze","internal_authored"]))
    return dict(version="routing-holdout-v2",scope="升级后的40题内部合成验收；作者参与实现，不是外部独立评测。",cases=cases)


if __name__=="__main__":
    path=Path(__file__).with_name("memory_routing_holdout.json")
    if path.exists():
        raise FileExistsError("禁止覆盖已存在验收集")
    path.write_text(json.dumps(build(),ensure_ascii=False,indent=2)+"\n",encoding="utf-8")
