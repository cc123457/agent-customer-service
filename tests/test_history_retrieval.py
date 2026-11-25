"""真实SQL/Chroma验证历史过滤、BM25/语义/RRF、完整问答、冷档案与上下文边界。"""
import asyncio
from datetime import datetime
from unittest.mock import Mock
from uuid import uuid4

import httpx
import pytest
from fastapi import FastAPI
from sqlalchemy import update
from sqlalchemy.exc import SQLAlchemyError

from db.memory_schema import conversations, messages
from memory.archive import MemoryArchive
from memory.budget import message_tokens
from memory.context_builder import compose_context
from memory.history_index import HistorySemanticIndex, turn_fingerprint
from memory.history_retrieval import BUSINESS_TZ, HistoryArchiveError, HistoryRetriever, history_query
from services.chat_service import ChatService
from tests.memory_helpers import complete, service


def write_round(repository, question, answer, session="old", user="1001", at=None):
    """用真实租约和事务写入问答；可固定业务时间，不调用聊天模型。"""
    old_clock = repository.clock
    if at is not None:
        repository.clock = lambda: at
    scope = ChatService.session_scope(user, session)
    owner = uuid4().hex
    assert repository.claim(scope, user, session, owner, 120)
    try:
        turn_id, _ = repository.start_request(scope, user, owner, uuid4().hex, question, [])
        repository.finish_request(scope, owner, turn_id, answer)
        return turn_id
    finally:
        repository.release(scope, owner)
        repository.clock = old_clock


class ConceptEmbeddings:
    """可控同义概念向量验证实际Chroma，不冒充真实Embedding召回质量。"""

    def __init__(self):
        """记录实际索引文本，用于确认用户/时间过滤发生在向量构建之前。"""
        self.documents = []

    def embed_query(self, text):
        """没有共享关键词的表达也可落入同一概念方向。"""
        concepts = (("轰隆", "睡眠", "小孩", "打扰"), ("充电", "底座", "电量", "补能"), ("木地板", "拖地", "擦洗"))
        position = next((index for index, words in enumerate(concepts) if any(word in text for word in words)), 3)
        return [float(index == position) for index in range(4)]

    def embed_documents(self, texts):
        """只为测试概念构造向量；数据库查询和向量索引不被模拟。"""
        self.documents.extend(texts)
        return [self.embed_query(text) for text in texts]


@pytest.fixture
def history_index(repository, tmp_path):
    """每个SQL后端配合独立真实Chroma，退出时释放Windows文件句柄。"""
    index = HistorySemanticIndex(repository, tmp_path / "history-index", ConceptEmbeddings(),
                                 chunk_chars=80, chunk_overlap=10)
    try:
        yield index
    finally:
        index.close()


def test_bm25_finds_assistant_resolution_and_returns_both_roles(repository):
    """关键词只出现在助手回答时仍能找回，结果带双方原文与来源。"""
    identity = write_round(repository, "机器不工作了", "清理充电触点后恢复，最终更换底座。")
    write_round(repository, "怎么拖地", "使用清水")
    retriever = HistoryRetriever(repository)
    found = retriever.search("1001", "最终更换底座的处理记录")
    assert found[0]["turn_id"] == identity
    assert [row["role"] for row in found[0]["messages"]] == ["user", "assistant"]
    assert found[0]["messages"][0]["content"] == "机器不工作了"
    assert "更换底座" in found[0]["messages"][1]["content"]
    assert found[0]["matched_by"] == ["bm25"]
    assert retriever.search("1001", "火星探测器") == []


def test_semantic_only_recall_and_no_plaintext_in_index(repository, history_index):
    """词面不重合时真实Chroma仍召回整轮，索引只保存向量和元数据。"""
    identity = write_round(repository, "机器半夜轰隆响吵醒小孩", "调整为白天运行。")
    retriever = HistoryRetriever(repository, semantic_index=history_index)
    query = "有没有谈过睡眠被打扰？"
    turns = repository.history_turns("1001")
    assert retriever._lexical(query, turns) == []
    found = retriever.search("1001", query)
    assert found[0]["turn_id"] == identity
    assert found[0]["matched_by"] == ["semantic"]
    assert "白天运行" in found[0]["messages"][1]["content"]
    stored = history_index.collection.get(include=["documents", "metadatas"])
    assert all(text is None for text in stored["documents"])
    before = len(history_index.embeddings.documents)
    retriever.search("1001", query)
    assert len(history_index.embeddings.documents) == before


def test_user_session_and_time_filters_apply_before_indexing(repository, history_index):
    """其他用户/会话/时间的数据不进入本次BM25和Embedding候选。"""
    allowed = write_round(repository, "充电底座故障", "清理触点", session="allowed", at=100)
    write_round(repository, "USER2_ONLY 充电底座故障", "其他用户秘密", user="1002", session="allowed", at=100)
    write_round(repository, "SESSION_ONLY 充电底座故障", "其他会话", session="other", at=100)
    write_round(repository, "TIME_ONLY 充电底座故障", "其他时间", session="allowed", at=200)
    retriever = HistoryRetriever(repository, semantic_index=history_index)
    found = retriever.search("1001", "充电底座", session_id="allowed", after=100, before=200)
    assert [turn["turn_id"] for turn in found] == [allowed]
    indexed = " ".join(history_index.embeddings.documents)
    assert not any(secret in indexed for secret in ("USER2_ONLY", "SESSION_ONLY", "TIME_ONLY"))
    assert retriever.search("1001", "充电底座", session_id="missing") == []


def test_candidate_window_is_applied_after_filters_and_excludes_recent_turns(repository):
    """候选上限不能先被其他用户/时间占满；近期窗口已有轮次不重复注入。"""
    old = write_round(repository, "充电座维修", "换底座", at=100)
    write_round(repository, "无关内容", "无关回答", at=200)
    write_round(repository, "其他用户充电座", "秘密", user="1002", at=300)
    retriever = HistoryRetriever(repository, max_turns=1)
    assert retriever.search("1001", "维修", before=200)[0]["turn_id"] == old
    assert retriever.search("1001", "维修", before=200, exclude_turn_ids=[old]) == []
    assert retriever.search("1001", "维修") == []  # 最旧轮次不在默认有限候选窗口。


def test_failed_and_incomplete_rounds_are_not_evidence(repository):
    """中断请求只有用户输入时，不能成为完整历史问答候选。"""
    scope, owner = ChatService.session_scope("1001", "failed"), uuid4().hex
    assert repository.claim(scope, "1001", "failed", owner, 120)
    repository.start_request(scope, "1001", owner, uuid4().hex, "底座故障秘密", [])
    repository.release(scope, owner)
    assert repository.history_turns("1001") == []
    assert HistoryRetriever(repository).search("1001", "底座") == []


def test_rrf_deduplicates_each_channel_by_turn_id():
    """同一问答多个块不会重复投票，多路命中才累加排名倒数。"""
    retriever = HistoryRetriever(Mock())
    ranked, scores = retriever._fuse([["a", "a", "b"], ["b", "b", "a"]])
    assert ranked == ["a", "b"]
    assert scores["a"] == scores["b"] == 1 / 61 + 1 / 62


def test_embedding_failure_keeps_bm25_and_sql_error_propagates(repository):
    """只降级向量通道，SQL故障仍抛错，不伪装成没有原文。"""
    identity = write_round(repository, "充电故障", "更换底座")
    index = Mock(search=Mock(side_effect=TimeoutError("embedding failed")))
    retriever = HistoryRetriever(repository, semantic_index=index)
    assert retriever.search("1001", "充电")[0]["turn_id"] == identity
    broken = HistoryRetriever(Mock(history_turns=Mock(side_effect=SQLAlchemyError("db failed"))))
    with pytest.raises(SQLAlchemyError):
        broken.search("1001", "充电")


def test_vector_candidates_are_rechecked_against_sql_snapshot(repository):
    """向量伪造轮次/指纹被丢弃；检索中途原文变化也不能输出旧索引结果。"""
    identity = write_round(repository, "半夜轰隆响", "白天运行")
    index = Mock()
    index.search.return_value = [{"turn_id": identity, "fingerprint": "old", "score": 1},
                                 {"turn_id": "other-user-turn", "fingerprint": "foreign", "score": 1}]
    retriever = HistoryRetriever(repository, semantic_index=index)
    assert retriever.search("1001", "睡眠") == []

    def change_after_recall(user_id, query, turns):
        """用真实SQL制造读取期间的变化，核对最终授权读取而非索引正文。"""
        fingerprint = turn_fingerprint(turns[0])
        with repository.engine.begin() as connection:
            connection.execute(update(messages).where(messages.c.turn_id == identity, messages.c.role == "assistant")
                               .values(content="新的原答"))
        return [{"turn_id": identity, "fingerprint": fingerprint, "score": 1}]

    index.search.side_effect = change_after_recall
    assert retriever.search("1001", "睡眠") == []


def test_cold_archive_is_hydrated_and_retirement_or_corruption_is_respected(repository, tmp_path):
    """冷热转换不丢检索能力；损坏明确报错，到期原文不能凭旧向量重现。"""
    identity = write_round(repository, "充电座坏了", "更换底座后恢复", at=100)
    turn = repository.history_turns("1001")[0]
    with repository.engine.begin() as connection:
        connection.execute(update(conversations).values(summary_upto=turn["messages"][-1]["id"]))
    archive = MemoryArchive(repository, tmp_path / "archives")
    report = archive.archive(200, apply=True)
    assert report["eligible_messages"] == 2
    retriever = HistoryRetriever(repository, archive=archive)
    found = retriever.search("1001", "底座")
    assert found[0]["turn_id"] == identity
    assert all(row.get("archived") for row in found[0]["messages"])
    path = next((tmp_path / "archives").glob("*.jsonl.gz"))
    original = path.read_bytes()
    path.write_bytes(b"corrupt")
    with pytest.raises(HistoryArchiveError):
        retriever.search("1001", "底座")
    path.write_bytes(original)
    archive.retire_archives(repository.clock() + 1, apply=True)
    assert retriever.search("1001", "底座") == []


def test_history_routing_dates_and_business_timezone():
    """按UTC+8解释相对日期、明确日期区间和当前会话，不把问候/更正当历史问题。"""
    now = datetime(2026, 11, 24, 12, tzinfo=BUSINESS_TZ).timestamp()
    plan = history_query("昨天充电座最后怎么处理的", "session-a", now)
    assert plan.after == datetime(2026, 11, 23, tzinfo=BUSINESS_TZ).timestamp()
    assert plan.before == datetime(2026, 11, 24, tzinfo=BUSINESS_TZ).timestamp()
    assert plan.preserve_original
    plan = history_query("当前会话2026-11-20到2026-11-22的充电记录", "session-a", now)
    assert plan.session_id == "session-a"
    assert plan.before == datetime(2026, 11, 23, tzinfo=BUSINESS_TZ).timestamp()
    assert history_query("最近7天的故障记录", now=now).after == now - 7 * 86400
    assert history_query("底座闪灯最后怎么处理的", now=now) is not None
    for text in ("你好？", "我用的是X20"):
        assert history_query(text, now=now) is None
    for text in ("2026-02-30的记录", "最近0天的记录"):
        with pytest.raises(ValueError):
            history_query(text, now=now)


def test_context_preserves_both_roles_and_marks_budget_excerpts(repository):
    """很长的命中按双方分配预算，不能截成只有问题没有解决办法。"""
    write_round(repository, "底座问题" * 300, "更换底座解决" * 300)
    turn = repository.history_turns("1001")[0]
    recent = [{"id": 9999, "turn_id": "current", "role": "user", "content": "继续排查"}]
    context = compose_context(recent, "", [], [turn], 1200, archive_budget=900)
    text = str(context)
    assert "用户原话#" in text and "助手原答#" in text and "原文节选" in text
    assert "更换底座" in text and "来源会话:old" in text
    assert context[-1]["content"] == "继续排查" and message_tokens(context) <= 1200


def test_chat_retrieval_without_old_trigger_words_and_historical_values(repository):
    """聊天可回忆助手原答；明确问旧事实时保留历史值，普通问题以当前事实为准。"""
    async def run():
        """走完整聊天流程验证注入的助手原答，以及旧值与当前事实的用途区分。"""
        chat = service(repository)
        write_round(repository, "底座的灯一直闪", "清理触点并更换底座")
        result = await complete(chat, "new", "底座闪灯最后怎么处理的？")
        assert "清理触点" in str(result.messages) and "助手原答#" in str(result.messages)
        await complete(chat, "model-old", "我用的是X10")
        await complete(chat, "model-new", "实际型号是X20")
        normal = await complete(chat, "normal", "我的设备型号是什么？")
        assert "X20" in str(normal.messages) and "X10" not in str(normal.messages)
        historical = await complete(chat, "past", "我以前说过的X10型号是什么？")
        assert "X10" in str(historical.messages) and "不代表当前事实" in str(historical.messages)
    asyncio.run(run())


def test_date_only_listing_does_not_run_generic_semantic_search(repository):
    """只有日期没有主题时按范围读取问答，不用泛化向量猜测用户想找哪件事。"""
    identity = write_round(repository, "充电故障", "清理触点", at=100)
    index = Mock()
    retriever = HistoryRetriever(repository, semantic_index=index)
    found = retriever.search("1001", "昨天聊了什么？", after=100, before=200)
    assert found[0]["turn_id"] == identity and found[0]["matched_by"] == ["scope"]
    index.search.assert_not_called()


@pytest.mark.parametrize("enabled", [True, False])
def test_history_factory_configuration_and_lifecycle(repository, tmp_path, monkeypatch, enabled):
    """实际依赖工厂装配独立历史索引，开关关闭时不加载向量组件。"""
    import sys
    from types import SimpleNamespace
    import api.deps
    from utils.config_handler import memory_conf
    monkeypatch.setattr(api.deps, "get_history_repository", lambda: repository)
    monkeypatch.setitem(sys.modules, "model.factory", SimpleNamespace(embed_model=ConceptEmbeddings()))
    monkeypatch.setitem(memory_conf, "history_retrieval", {
        "enabled": enabled, "max_turns": 10,
        "semantic": {"enabled": True, "directory": str(tmp_path / "factory-history")},
    })
    retriever = api.deps.get_history_retriever.__wrapped__()
    try:
        assert retriever.enabled == enabled and retriever.max_turns == 10
        assert (retriever.semantic_index is not None) == enabled
    finally:
        retriever.close()


def test_history_api_filters_authentication_and_errors(repository):
    """API强制JWT身份，显式时间/会话可过滤，故障响应不包含内部路径或数据库信息。"""
    from api.deps import get_current_user, get_history_retriever, get_limited_user
    from api.memory import router
    from utils.security import create_token
    write_round(repository, "充电座闪灯", "清理触点", session="one", at=100)
    write_round(repository, "充电座闪灯", "别人秘密", user="1002", session="one", at=100)
    retriever = HistoryRetriever(repository)
    app = FastAPI()
    app.include_router(router)
    app.dependency_overrides[get_history_retriever] = lambda: retriever
    app.dependency_overrides[get_limited_user] = get_current_user

    async def run():
        """使用真实ASGI与JWT，测试不向真实模型发送问题。"""
        async with httpx.AsyncClient(transport=httpx.ASGITransport(app=app), base_url="http://test") as client:
            assert (await client.get("/history/search", params={"query": "充电"})).status_code in (401, 403)
            headers = {"Authorization": "Bearer " + create_token("1001")}
            response = await client.get("/history/search", params={"query": "充电", "session_id": "one",
                                        "after": "1970-01-01T00:01:40Z", "before": "1970-01-01T00:03:20Z"}, headers=headers)
            assert response.status_code == 200 and len(response.json()["turns"]) == 1
            assert "清理触点" in response.text and "别人秘密" not in response.text
            invalid = await client.get("/history/search", params={"query": "充电", "after": "2026-11-24", "before": "2026-11-23"}, headers=headers)
            assert invalid.status_code == 422
            retriever.search = Mock(side_effect=HistoryArchiveError("private-path"))
            failed = await client.get("/history/search", params={"query": "充电"}, headers=headers)
            assert failed.status_code == 503 and "private-path" not in failed.text
    asyncio.run(run())
