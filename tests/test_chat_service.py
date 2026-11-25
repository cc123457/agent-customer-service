"""唯一聊天服务的预算、摘要失败、超时和取消边界，使用当前SQL主链路验证。"""
import asyncio
from types import SimpleNamespace

import pytest
from memory.session_lease import SessionLeaseStore
from services.chat_service import ChatService
from services.chat_types import CompressionStatus, ContextBudgetError
from tests.memory_helpers import service, complete


def test_oversized_input_rejected_before_archive(repository):
    async def run():
        chat = service(repository, context_budget=200, summary_budget=50)
        with pytest.raises(ContextBudgetError):
            await chat.open_turn("oversized", "字" * 100, "1001")
        assert repository.history(chat.session_scope("1001", "oversized"), "1001") == []
        assert chat.leases.claim(chat.session_scope("1001", "oversized"), "inspect", 10)
    asyncio.run(run())


@pytest.mark.parametrize("failure", ["timeout", "empty", "oversized", "commit"])
def test_failed_summary_keeps_old_watermark_and_originals(repository, failure):
    async def run():
        chat = service(repository, summary_timeout=0.01)
        await complete(chat, "failure", "第一轮")
        await complete(chat, "failure", "第二轮")
        old_commit = repository.commit_summary
        if failure == "timeout":
            async def delayed(prompt):
                """可取消长等待，只用于触发单批摘要超时。"""
                await asyncio.sleep(1)
            chat._summarizer.ainvoke = delayed
        elif failure == "empty":
            chat._summarizer.ainvoke.return_value = SimpleNamespace(content="  ")
        elif failure == "oversized":
            chat._summarizer.ainvoke.return_value = SimpleNamespace(content="字" * 400)
        else:
            def broken_commit(*args):
                """模拟提交故障，验证生成成功也不会先删除原文。"""
                raise RuntimeError("写入失败")
            repository.commit_summary = broken_commit
        try:
            turn = await complete(chat, "failure", "第三轮")
        finally:
            repository.commit_summary = old_commit
        assert turn.compression == CompressionStatus.FAILED
        assert len(repository.history(turn.scope, "1001")) == 6
        assert repository.claim(turn.scope, "1001", "failure", "inspect", 120)
        snapshot = repository.snapshot(turn.scope, "inspect")
        assert snapshot["summary"] == "" and snapshot["summary_upto"] == 0
        repository.release(turn.scope, "inspect")
    asyncio.run(run())


def test_turn_timeout_marks_failed_and_releases_both_leases(repository):
    class SlowAgent:
        async def execute_stream(self, messages, user_id):
            """在生成阶段等待至超时，不返回伪造答案。"""
            await asyncio.sleep(1)
            yield {"type": "answer", "content": "迟到回答"}

    async def run():
        chat = ChatService(SlowAgent(), SessionLeaseStore(), repository, turn_timeout=0.1)
        turn = await chat.open_turn("timeout", "问题", "1001")
        with pytest.raises(TimeoutError):
            async for _ in chat.reply_turn(turn, "1001"):
                pass
        assert not chat.leases.owns(turn.scope, turn.owner)
        assert repository.history(turn.scope, "1001")[0]["status"] == "failed"
        assert repository.claim(turn.scope, "1001", "timeout", "next", 120)
        repository.release(turn.scope, "next")
    asyncio.run(run())


def test_cancelled_reply_releases_lease_and_preserves_user_record(repository):
    async def run():
        entered = asyncio.Event()

        class PendingAgent:
            async def execute_stream(self, messages, user_id):
                """通知测试已进入生成，再等待外部取消。"""
                entered.set()
                await asyncio.Event().wait()
                yield

        chat = ChatService(PendingAgent(), SessionLeaseStore(), repository)
        turn = await chat.open_turn("cancel", "问题", "1001")

        async def consume():
            """消费生成器，确保取消发生在Agent执行过程中。"""
            async for _ in chat.reply_turn(turn, "1001"):
                pass

        task = asyncio.create_task(consume())
        await entered.wait()
        task.cancel()
        with pytest.raises(asyncio.CancelledError):
            await task
        assert not chat.leases.owns(turn.scope, turn.owner)
        assert repository.history(turn.scope, "1001")[0]["status"] == "failed"
    asyncio.run(run())
