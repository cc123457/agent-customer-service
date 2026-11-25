"""记忆升级回归：真实关系库和Chroma，向量/模型可控以稳定验证失败、纠正、隔离和归档。"""
import asyncio
import json
from types import SimpleNamespace
from unittest.mock import AsyncMock, Mock

import pytest
from sqlalchemy import select

from memory.archive import MemoryArchive
from memory.request_analyzer import MemoryRequestAnalyzer
from memory.long_term import recall_memories
from memory.semantic_index import SemanticMemoryIndex
from services.chat_types import SessionCapacityError
from tests.memory_helpers import service, complete


class ConceptEmbeddings:
    """可控同义词向量，仅验证真实Chroma检索和索引一致性，不代表真实Embedding效果。"""

    def embed_query(self, text):
        """为三个独立测试概念构造正交向量，避免随机向量导致断言不稳定。"""
        for index, words in enumerate((("噪声", "安静"), ("预算", "便宜"), ("触点", "排查"))):
            if any(word in text for word in words):
                return [float(index == position) for position in range(4)]
        return [0., 0., 0., 1.]

    def embed_documents(self, texts):
        """索引文本与查询使用同一可控向量空间。"""
        return [self.embed_query(text) for text in texts]


def test_semantic_recall_and_sql_delete_validation(repository, tmp_path):
    async def run():
        chat = service(repository)
        await complete(chat, "notes", "记住：我对噪声非常敏感。")
        index = SemanticMemoryIndex(repository, tmp_path / "index", ConceptEmbeddings())
        index.sync_user("1001")
        found = recall_memories(repository, "1001", "推荐安静的机器人", [], index)
        assert any("噪声" in row["value"] for row in found)
        assert index.search("1002", "安静") == []
        note = repository.list_memories("1001")[0]
        repository.delete_memory("1001", note["id"])
        # 故意不更新索引，SQL复核仍必须阻止已删除内容返回。
        assert index.search("1001", "安静") == []
        index.sync_user("1001")
        assert index.collection.get(where={"user_id": "1001"})["ids"] == []
        index.close()
    asyncio.run(run())


def test_stale_vector_not_used_after_correction(repository, tmp_path):
    async def run():
        chat = service(repository)
        await complete(chat, "first", "我用的是X10")
        index = SemanticMemoryIndex(repository, tmp_path / "index", ConceptEmbeddings())
        index.sync_user("1001")
        await complete(chat, "second", "实际型号是B8")
        assert index.search("1001", "型号") == []
        index.sync_user("1001")
        assert index.search("1001", "型号")[0]["value"] == "B8"
        index.close()
    asyncio.run(run())


def test_two_devices_and_conflict_clarification(repository):
    async def run():
        chat = service(repository)
        await complete(chat, "two", "客厅的机器人型号是X10，卧室的机器人型号是B8。")
        both = recall_memories(repository, "1001", "我的设备型号是什么？", [])
        assert {"X10", "B8"}.issubset({row["value"] for row in both})
        assert any(row.get("needs_clarification") for row in both)
        await complete(chat, "correct", "客厅的机器人型号更正为X20。")
        living = recall_memories(repository, "1001", "客厅的设备型号是什么？", [])
        assert "X20" in str(living) and "X10" not in str(living) and "B8" not in str(living)
        await complete(chat, "conflict", "我的设备型号是Q1。我的设备型号是Q2。")
        rows = recall_memories(repository, "1001", "我的设备型号是什么？", [])
        assert any(row.get("needs_clarification") for row in rows)
        assert not any(row["value"] in ("Q1", "Q2") for row in rows)
    asyncio.run(run())


def test_fact_model_validates_structure_and_original_evidence():
    async def run():
        text = "我一直偏爱安静的机器。"
        good = {"facts": [{"key": "note", "value": "偏爱安静的机器", "entity": "profile",
                            "evidence": text, "valid_at": "", "action": "upsert"}]}
        bad = {"facts": [{**good["facts"][0], "value": "不存在的事实"}]}
        model = Mock()
        model.bind.return_value = model
        model.ainvoke = AsyncMock(side_effect=[SimpleNamespace(content=json.dumps(bad)), SimpleNamespace(content=json.dumps(good))])
        rows = await MemoryRequestAnalyzer(model).extract(text)
        assert rows == []
        assert model.ainvoke.await_count == 1
        model.ainvoke = AsyncMock(return_value=SimpleNamespace(content=json.dumps(good)))
        assert (await MemoryRequestAnalyzer(model).extract(text))[0].value == "偏爱安静的机器"
        model.ainvoke.reset_mock()
        assert await MemoryRequestAnalyzer(model).extract("如果我用的是X10会怎样？") == []
        assert model.ainvoke.await_count == 0
        assert (await MemoryRequestAnalyzer(model).extract("我用的是X10"))[0].value == "X10"
        assert model.ainvoke.await_count == 0
    asyncio.run(run())


def test_fact_model_failure_keeps_conversation_available():
    async def run():
        model = Mock()
        model.bind.return_value = model
        model.ainvoke = AsyncMock(side_effect=TimeoutError("接口超时"))
        assert await MemoryRequestAnalyzer(model).extract("我一直偏爱安静的机器") == []
    asyncio.run(run())


def test_valid_json_with_wrong_business_field_is_rejected():
    async def run():
        text = "我对机器运转时的声音特别敏感。"
        wrong = {"facts": [{"key": "reply_style", "value": "声音特别敏感", "entity": "profile",
                             "evidence": text, "valid_at": "", "action": "upsert"}]}
        model = Mock()
        model.bind.return_value = model
        model.ainvoke = AsyncMock(return_value=SimpleNamespace(content=json.dumps(wrong)))
        assert await MemoryRequestAnalyzer(model).extract(text) == []
        assert model.ainvoke.await_count == 1
    asyncio.run(run())


def test_model_cannot_overwrite_two_conflicting_values_in_one_batch():
    async def run():
        text = "我正在用的那台是X10，同时标注为B8。"
        facts = {"facts": [{"key": "device_model", "value": value, "entity": "profile",
                             "evidence": text, "valid_at": "", "action": "upsert"} for value in ("X10", "B8")]}
        model = Mock()
        model.bind.return_value = model
        model.ainvoke = AsyncMock(return_value=SimpleNamespace(content=json.dumps(facts)))
        rows = await MemoryRequestAnalyzer(model).extract(text)
        assert len(rows) == 1 and rows[0].action == "conflict"
    asyncio.run(run())


def test_archive_export_hydrate_checksum_and_expiry(repository, tmp_path):
    async def run():
        repository.clock = Mock(return_value=100)
        chat = service(repository)
        for text in ("早期原话", "第二轮", "第三轮"):
            turn = await complete(chat, "archive", text)
        old = repository.history(turn.scope, "1001")
        repository.clock.return_value = 100 + 100 * 86400
        archive = MemoryArchive(repository, tmp_path / "archives")
        preview = archive.archive(100 + 10 * 86400)
        assert preview["eligible_messages"] == 2 and not preview["applied"]
        assert not archive.directory.exists()
        result = archive.archive(100 + 10 * 86400, apply=True)
        assert result["applied"]
        current = repository.history(turn.scope, "1001")
        assert sum(row["content"] == "" for row in current) == 2
        restored = archive.hydrate(current)
        assert [row["content"] for row in restored] == [row["content"] for row in old]
        assert archive.archive(repository.clock(), apply=True)["eligible_messages"] == 0
        path = next(archive.directory.glob("*.gz"))
        original = path.read_bytes()
        path.write_bytes(b"corrupted")
        with pytest.raises(ValueError):
            archive.hydrate(current)
        path.write_bytes(original)
        report = archive.retire_archives(repository.clock() + 1, apply=True)
        assert report["expired_archive_files"] == 1 and not path.exists()
        assert any(row.get("expired") for row in archive.hydrate(current))
    asyncio.run(run())


def test_active_conversation_not_archived(repository, tmp_path):
    async def run():
        repository.clock = Mock(return_value=100)
        chat = service(repository)
        for text in ("第一轮", "第二轮", "第三轮"):
            turn = await complete(chat, "busy", text)
        repository.clock.return_value = 1000
        pending = await chat.open_turn("busy", "处理中", "1001")
        archive = MemoryArchive(repository, tmp_path / "archive")
        assert archive.archive(2000, apply=True)["eligible_messages"] == 0
        await chat.close_turn(pending)
    asyncio.run(run())


def test_quota_failure_rolls_back_new_request(repository):
    async def run():
        chat = service(repository)
        repository.limits["max_hot_bytes_per_user"] = 50
        turn = await complete(chat, "quota", "短句")
        with pytest.raises(SessionCapacityError):
            await chat.open_turn("quota", "长" * 100, "1001")
        assert len(repository.history(turn.scope, "1001")) == 2
        repository.limits["max_hot_bytes_per_user"] = 10000000
        repository.limits["max_active_notes_per_user"] = 1
        await complete(chat, "notes", "记住：第一条笔记")
        with pytest.raises(SessionCapacityError):
            await chat.open_turn("notes", "记住：第二条笔记", "1001")
        assert len(repository.list_memories("1001")) == 1
    asyncio.run(run())


def test_fixed_memory_development_cases_with_controlled_embeddings(repository, tmp_path):
    from pathlib import Path
    from scripts.evaluate_memory import evaluate
    cases = json.loads(Path("evals/memory_cases.json").read_text(encoding="utf-8"))
    assert len({row["id"] for row in cases}) == len(cases) == 30
    index = SemanticMemoryIndex(repository, tmp_path / "evaluation-index", ConceptEmbeddings())
    try:
        result = evaluate(repository, cases, index)
        assert result["passed"] == 30, [row for row in result["results"] if not row["passed"]]
        assert result["matched_facts"] == 20 and result["wrong_recall_cases"] == 0
    finally:
        index.close()


def test_known_superseded_value_redacted_from_old_summary(repository):
    from memory.conflict_resolution import sanitize_reference

    async def run():
        chat = service(repository)
        await complete(chat, "old", "我用的是X10")
        await complete(chat, "new", "实际型号是B8")
        old = repository.superseded_values("1001")
        assert "X10" in old
        sanitized = sanitize_reference("以前使用X10，文档中还有X100。", old)
        assert "[该历史值已更正或停用]" in sanitized and "X100" in sanitized
        assert repository.superseded_values("1002") == []
    asyncio.run(run())


def test_version_and_whole_checkpoint_retention(repository, tmp_path):
    from db.memory_schema import summary_versions
    from memory.checkpointer import SqlCheckpointSaver
    from langgraph.checkpoint.base import empty_checkpoint

    async def run():
        repository.clock = Mock(return_value=100)
        chat = service(repository)
        for index in range(7):
            turn = await complete(chat, "versions", f"第{index}轮")
        archive = MemoryArchive(repository, tmp_path / "archives")
        preview = archive.prune_versions(keep=1)
        assert preview["obsolete_summary_versions"] > 0
        archive.prune_versions(keep=1, apply=True)
        with repository.engine.connect() as connection:
            assert len(connection.execute(select(summary_versions)).all()) == 1
        saver = SqlCheckpointSaver(repository.engine)
        config = saver.put({"configurable": {"thread_id": turn.turn_id}}, empty_checkpoint(), {"source": "input", "step": 0, "parents": {}}, {})
        preview = archive.prune_checkpoints(200, 200)
        assert preview["expired_checkpoint_threads"] == 1 and saver.get_tuple(config)
        archive.prune_checkpoints(200, 200, apply=True)
        assert saver.get_tuple(config) is None
        assert len(repository.history(turn.scope, "1001")) == 14
    asyncio.run(run())
