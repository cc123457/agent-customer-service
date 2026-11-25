"""小规模可复现验证：原文归档、真实分批摘要、跨会话记忆召回；--live才调用模型。"""
import argparse
import asyncio
import json
import os
import sys
import tempfile
from pathlib import Path
from types import SimpleNamespace
from uuid import uuid4

from dotenv import load_dotenv

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))


async def verify(repository, agent, summary_model):
    """执行三轮触发小预算分批摘要，再换会话检查召回；数据库不删除原文，不接触其他会话。"""
    from memory.session_lease import SessionLeaseStore
    from services.chat_service import ChatService

    class CountedSummary:
        def __init__(self):
            """初始化本次摘要调用计数，与供应商HTTP重试次数不同。"""
            self.calls = 0

        async def ainvoke(self, prompt):
            """透传摘要调用，供结果报告证明本次确实分成多个请求。"""
            self.calls += 1
            return await summary_model.ainvoke(prompt)

    counted = CountedSummary()
    service = ChatService(
        agent, SessionLeaseStore(), repository, history_window=4, summarizer=counted,
        summary_input_budget=1200, summary_budget=500,
    )
    session = "verify-" + uuid4().hex
    seed = "我用的是X10，我家是木地板。\n" + (
        "维护记录：已确认插座通电，等待后续检查，不要重复假定电源故障。\n" * 12
    ) + "请只回复收到，不要分析或调用工具。"
    turns = []
    for text in (seed, "请只回复收到。", "继续记录，请只回复收到。"):
        turn = await service.open_turn(session, text, "1001")
        async for _ in service.reply_turn(turn, "1001"):
            pass
        turns.append(turn)
    followup = await service.open_turn(
        session + "-new", "不用查产品知识，只说之前记录的设备型号和家里地面。", "1001",
    )
    assert "X10" in str(followup.messages) and "木地板" in str(followup.messages)
    answers = []
    async for event in service.reply_turn(followup, "1001"):
        if event["type"] == "answer":
            answers.append(event["content"])
    originals = repository.history(turns[0].scope, "1001")
    assert len(originals) == 6 and any(row["content"] == seed for row in originals)
    assert counted.calls > 1
    return {
        "backend": repository.engine.dialect.name,
        "compression_status": [turn.compression.value for turn in turns],
        "summary_invocations": counted.calls,
        "original_records_after_compression": len(originals),
        "original_seed_preserved": True,
        "cross_session_context_has_model_and_floor": True,
        "followup_answer": "".join(answers),
        "limitations": "单组功能验证，不是准确率、性能或长期记忆保真度基准。",
    }


def main():
    """读取选定配置；默认临时SQLite/模型替身，--live使用真实模型并产生费用；输出不含凭据。"""
    parser = argparse.ArgumentParser(description="记忆全链路小规模验证")
    parser.add_argument("--live", action="store_true")
    parser.add_argument("--env-file", type=Path)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    if args.env_file:
        if not args.env_file.is_file():
            parser.error("环境文件不存在")
        load_dotenv(args.env_file, override=True)
    os.environ["LANGFUSE_PUBLIC_KEY"] = ""
    os.environ["LANGFUSE_SECRET_KEY"] = ""
    from db.memory_repository import MemoryRepository

    class StubAgent:
        async def execute_stream(self, messages, user_id):
            """离线模式只验证程序管线，固定返回文字，不冒充模型真实推理。"""
            yield {"type": "answer", "content": "收到。"}

    class StubSummary:
        async def ainvoke(self, prompt):
            """离线模式的可控摘要替身，不调用外部服务。"""
            return SimpleNamespace(content="用户使用X10，木地板；已确认插座通电。")

    with tempfile.TemporaryDirectory(prefix="agent-memory-") as directory:
        url = os.getenv("MEMORY_DATABASE_URL") or "sqlite:///" + (Path(directory) / "memory.db").as_posix()
        repository = MemoryRepository(url)
        repository.initialize()
        try:
            if args.live:
                from agent.react_agent import ReactAgent
                from memory.checkpointer import SqlCheckpointSaver
                from model.factory import chat_model
                agent, summary_model = ReactAgent(checkpointer=SqlCheckpointSaver(repository.engine)), chat_model
            else:
                agent, summary_model = StubAgent(), StubSummary()
            result = asyncio.run(verify(repository, agent, summary_model))
            result["real_model"] = args.live
        finally:
            repository.engine.dispose()
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
