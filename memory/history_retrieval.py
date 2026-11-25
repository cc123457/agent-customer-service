"""历史检索：受控时间/会话过滤，BM25与向量按轮次RRF融合，再从SQL恢复双方原文。"""
import logging
import math
import re
import time
from collections import defaultdict
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone

import jieba
from rank_bm25 import BM25Plus

from memory.history_index import turn_fingerprint


logger = logging.getLogger(__name__)
BUSINESS_TZ = timezone(timedelta(hours=8))
STOP_WORDS = frozenset({"什么", "怎么", "如何", "一下", "这个", "那个", "之前", "上次", "请问", "我的", "我们",
                        "以前", "历史", "那次", "当时", "后来", "最后", "昨天", "今天", "前天", "记录", "会话",
                        "帮我", "查询", "查找", "回顾", "记得", "说过", "提过", "聊过", "聊了", "聊天",
                        "内容", "哪些", "问题", "解决", "处理"})


class HistoryArchiveError(RuntimeError):
    """历史冷原文不可验证或不可读取，不能将此故障伪装成没有历史。"""


@dataclass(frozen=True)
class HistoryQuery:
    """确定性解析后的查询范围；user_id始终由后端另行传入，不能从问题中指定。"""
    text: str
    session_id: str | None = None
    after: float | None = None
    before: float | None = None
    preserve_original: bool = False


def history_query(text, current_session=None, now=None):
    """识别历史/问句意图及有限中文日期；普通陈述和问候不额外执行历史检索。

    支持今天/昨天/前天/最近N天/明确YYYY-MM-DD日期，以及当前会话。
    未解析的时间表达不猜测；上层显式检索API也可以直接传入精确时间范围。
    """
    if re.fullmatch(r"(?:你好|您好|嗨|谢谢|好的|收到)[!！?？。\s]*", text.strip()):
        return None
    if not re.search(r"上次|以前|之前|历史|那次|当时|后来|最后|还记得|聊过|说过|提过|昨天|前天|今天|最近\d+天|\d{4}-\d{2}-\d{2}|[?？]|怎么|如何|为什么|什么|能否|是否|办法|方案|处理|解决", text):
        return None
    return parse_history_scope(text, current_session, now)


def parse_history_scope(text, current_session=None, now=None):
    """只解析范围，不决定是否搜索；非法日期明确报错，不能悄悄扩大查询范围。"""
    moment = datetime.fromtimestamp(time.time() if now is None else now, BUSINESS_TZ)
    midnight = moment.replace(hour=0, minute=0, second=0, microsecond=0)
    after = before = None
    days_ago = next((days for word, days in (("前天", 2), ("昨天", 1), ("今天", 0)) if word in text), None)
    recent = re.search(r"最近(\d{1,3})天", text)
    dates = re.findall(r"\d{4}-\d{2}-\d{2}", text)
    if dates:
        try:
            parsed = [datetime.strptime(value, "%Y-%m-%d").replace(tzinfo=BUSINESS_TZ) for value in dates[:2]]
        except ValueError as error:
            raise ValueError("历史查询日期无效，请使用真实的YYYY-MM-DD日期") from error
        after = parsed[0].timestamp()
        before = (parsed[-1] + timedelta(days=1)).timestamp()
        if after >= before:
            raise ValueError("历史查询结束日期不能早于开始日期")
    elif recent:
        days = int(recent.group(1))
        if days < 1:
            raise ValueError("最近天数必须大于0")
        after, before = (moment - timedelta(days=days)).timestamp(), moment.timestamp()
    elif days_ago is not None:
        start = midnight - timedelta(days=days_ago)
        after, before = start.timestamp(), (start + timedelta(days=1)).timestamp()
    session = current_session if re.search(r"当前会话|本次会话|这个会话", text) else None
    explicit_history = bool(after is not None or re.search(
        r"上次|以前|之前|历史|那次|当时|原来|曾经|用过|还记得|聊过|说过|提过", text,
    ))
    return HistoryQuery(text, session, after, before, explicit_history)


def tokenize_history(text):
    """中文分词并保留型号/错误码；排除检索指令词，不把零匹配文档作为BM25命中。"""
    return [word.lower() for word in jieba.lcut(text[:100000])
            if len(word.strip()) >= 2 and word not in STOP_WORDS and re.search(r"[\w\u4e00-\u9fff]", word)]


class HistoryRetriever:
    """有界候选上的历史混合检索；共享组件不保存某个用户的上次查询结果。"""

    def __init__(self, repository, semantic_index=None, archive=None, enabled=True,
                 max_turns=200, candidate_limit=12, result_limit=3, rrf_k=60):
        """组装SQL、可选向量索引及冷档案读取器；结果数量和候选窗口均有配置上限。"""
        if min(max_turns, candidate_limit, result_limit, rrf_k) < 1 or result_limit > candidate_limit:
            raise ValueError("历史检索数量与RRF配置无效")
        self.repository = repository
        self.semantic_index = semantic_index
        self.archive = archive
        self.enabled = enabled
        self.max_turns = max_turns
        self.candidate_limit = candidate_limit
        self.result_limit = result_limit
        self.rrf_k = rrf_k

    def _hydrate(self, turns):
        """批量恢复冷原文；到期/不完整问答跳过，损坏或缺失冷文件明确向上传播。"""
        originals = [row for turn in turns for row in turn["messages"]]
        try:
            hydrated = self.archive.hydrate(originals) if self.archive else originals
        except (OSError, ValueError, KeyError) as error:
            raise HistoryArchiveError("历史冷档案暂不可读取") from error
        by_id = {row["id"]: row for row in hydrated}
        result = []
        for turn in turns:
            pair = [dict(by_id[row["id"]]) for row in turn["messages"]]
            if any(row.get("expired") or not row["content"].strip() for row in pair):
                continue
            result.append(dict(turn, messages=pair))
        return result

    def _lexical(self, query, turns):
        """在过滤后的整轮问答上运行BM25Plus，明确排除没有任何查询词重叠的文档。"""
        tokens = tokenize_history(query)
        documents = [tokenize_history("\n".join(row["content"] for row in turn["messages"])) for turn in turns]
        if not tokens or not any(documents):
            return []
        # 少量空文档仍保留位置，避免轮次ID与评分错位。
        model = BM25Plus([words or ["__empty_history__"] for words in documents])
        scores = model.get_scores(tokens)
        ranked = [(turn["turn_id"], float(score)) for turn, words, score in zip(turns, documents, scores)
                  if set(tokens).intersection(words)]
        ranked.sort(key=lambda row: row[1], reverse=True)
        return [identity for identity, _ in ranked[:self.candidate_limit]]

    def _fuse(self, channels):
        """先在每路按轮次去重，再累加RRF分数；一个问答命中多个向量块也只投一票。"""
        scores = defaultdict(float)
        for channel in channels:
            for rank, identity in enumerate(dict.fromkeys(channel), 1):
                scores[identity] += 1 / (self.rrf_k + rank)
        return sorted(scores, key=lambda identity: scores[identity], reverse=True), scores

    def search(self, user_id, query, *, session_id=None, after=None, before=None, exclude_turn_ids=()):
        """过滤→完整问答→BM25/语义→RRF→SQL再授权复核；返回原文，不返回向量库正文。"""
        if not self.enabled or not query.strip():
            return []
        if any(value is not None and not math.isfinite(value) for value in (after, before)):
            raise ValueError("历史检索时间必须是有限数值")
        filters = dict(session_id=session_id, after=after, before=before, exclude_turn_ids=exclude_turn_ids)
        turns = self._hydrate(self.repository.history_turns(user_id, limit=self.max_turns, **filters))
        allowed = self.repository.access_policy(user_id).turns(turns)
        if self.semantic_index is not None and hasattr(self.semantic_index, "remove_turns"):
            allowed_ids = {turn["turn_id"] for turn in allowed}
            blocked = [turn for turn in turns if turn["turn_id"] not in allowed_ids]
            try:
                self.semantic_index.remove_turns(user_id, blocked)
            except Exception:
                logger.warning("旧历史向量清理失败，SQL遗忘过滤仍然生效", exc_info=True)
        turns = allowed
        if not turns:
            return []
        fingerprints = {turn["turn_id"]: turn_fingerprint(turn) for turn in turns}
        terms_query = re.sub(r"\d{4}-\d{2}-\d{2}|最近\d{1,3}天", "", query)
        scoped_listing = not tokenize_history(terms_query) and (after is not None or before is not None or session_id is not None)
        lexical = self._lexical(query, turns) if not scoped_listing else []
        semantic = []
        if self.semantic_index is not None and not scoped_listing:
            try:
                found = self.semantic_index.search(user_id, query, turns)
                semantic = [row["turn_id"] for row in found
                            if fingerprints.get(row["turn_id"]) == row["fingerprint"]]
            except Exception:
                logger.warning("历史语义检索不可用，继续使用BM25关键词召回", exc_info=True)
        scoped = [turn["turn_id"] for turn in turns[:self.candidate_limit]] if scoped_listing else []
        ranked, scores = self._fuse([lexical, semantic, scoped])
        if not ranked:
            return []
        fresh = self._hydrate(self.repository.history_turns(
            user_id, limit=len(ranked), turn_ids=ranked, **filters,
        ))
        fresh = self.repository.access_policy(user_id).turns(fresh)
        valid = {turn["turn_id"]: turn for turn in fresh
                 if turn_fingerprint(turn) == fingerprints.get(turn["turn_id"])}
        return [dict(valid[identity], score=scores[identity],
                     matched_by=[name for name, ids in (("bm25", lexical), ("semantic", semantic), ("scope", scoped)) if identity in ids])
                for identity in ranked if identity in valid][:self.result_limit]

    def close(self):
        """关闭已有历史向量客户端，不初始化额外组件。"""
        if self.semantic_index is not None:
            self.semantic_index.close()
