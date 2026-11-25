"""独立历史检索与四种上下文对照；默认离线BM25，--live才调用真实模型和Embedding。"""
import argparse
import asyncio
import hashlib
import json
import os
import sys
import tempfile
import time
from datetime import datetime
from pathlib import Path
from threading import Lock

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
ROOT = Path(__file__).resolve().parents[1]
from model.usage import UsageLedger, combine_reports, current_usage_stage, record_usage, usage_scope

from evals.history_metrics import (abstention_metrics, answer_score, percentile, retrieval_score, validate_dataset, visible_history)
from memory.budget import message_tokens
from memory.answer_policy import MEMORY_ANSWER_POLICY
from memory.context_builder import compose_context
from memory.history_retrieval import HistoryRetriever, history_query
from pydantic import BaseModel, ConfigDict, StrictBool, StrictStr
from services.json_payload import json_payload

ARMS = ("recent", "summary_recent", "memory", "full_history")
SYSTEM = """你在参加客服历史记忆测试。只依据提供的数据回答当前用户最后的问题。
区分当时原话与当前更正，当前事实采用提问时最新的明确更正；历史问题按指定时间回答。
已明确要求遗忘的值不能从旧原文恢复。没有证据就拒答，不猜编号、时间或私密信息。
仅输出一个JSON对象：{"answer":"简短答案，只含所问内容","abstain":false}。
资料不足或应当遗忘时abstain为true，answer简述缺少依据。不要复述无关信息或推理过程。
"""


class ShortAnswer(BaseModel):
    """在评测端严格校验回答结构；结构通过和事实得分分别统计。"""
    model_config = ConfigDict(extra="forbid")
    answer: StrictStr
    abstain: StrictBool


class SemanticOnly(HistoryRetriever):
    """仅为消融关闭词法一路，仍复用现有过滤、指纹复核、排序和TopK实现。"""
    def _lexical(self, query, turns):
        """返回空词法候选，让原检索器只对向量通道进行排序。"""
        return []


class CountedEmbeddings:
    """统计真实Embedding请求和输入字节；SDK未暴露用量时不伪造token数。"""
    def __init__(self, delegate):
        """包装已配置的真实Embedding对象，线程安全地保留成功/失败记录。"""
        self.delegate, self.events, self.lock = delegate, [], Lock()

    def _invoke(self, kind, texts):
        """执行一次SDK调用并记录耗时，错误重新抛出以保持生产降级行为。"""
        start, event = time.perf_counter(), {"kind": kind, "texts": len(texts),
                                           "input_bytes": sum(len(text.encode("utf-8")) for text in texts)}
        try:
            with usage_scope(stage=current_usage_stage() + "/" + kind):
                return (self.delegate.embed_documents(texts) if kind == "documents"
                        else self.delegate.embed_query(texts[0]))
        except Exception as error:
            event["error"] = type(error).__name__
            raise
        finally:
            event["seconds"] = time.perf_counter() - start
            with self.lock:
                self.events.append(event)

    def embed_documents(self, texts):
        """为文档索引生成真实向量，记录批次而非把每个文本冒充一次请求。"""
        return self._invoke("documents", texts)

    def embed_query(self, text):
        """查询也调用真实Embedding，不使用预置答案或概念向量。"""
        return self._invoke("query", [text])


class ProviderAccessBlocked(RuntimeError):
    """本次评测已遇到401/403，禁止继续批量调用；不伪装成模型返回空答案。"""


class RequestGate:
    """仅限制评测进程的模型请求节奏，不改变生产逻辑、模型参数或答案评分。"""
    def __init__(self, interval):
        """共享全局请求间隔；并行案例仍可执行SQL、向量和上下文操作。"""
        self.interval, self.next_start, self.lock = interval, 0, asyncio.Lock()
        self.blocked_status = None

    def block_access(self, status):
        """共享访问拒绝状态，避免免费额度耗尽后继续逐题发送无效请求。"""
        if status in (401, 403):
            self.blocked_status = status

    async def wait(self):
        """串行预留请求开始时刻，异步等待，返回评测人为排队时间。"""
        start = time.perf_counter()
        async with self.lock:
            if self.blocked_status:
                raise ProviderAccessBlocked(f"模型访问已被{self.blocked_status}拒绝，本轮后续调用停止")
            delay = max(0, self.next_start - time.perf_counter())
            if delay:
                await asyncio.sleep(delay)
            if self.blocked_status:
                raise ProviderAccessBlocked(f"模型访问已被{self.blocked_status}拒绝，本轮后续调用停止")
            self.next_start = time.perf_counter() + self.interval
        return time.perf_counter() - start


class CountedModel:
    """统计聊天/摘要的真实调用；Omni使用流式SDK并聚合为最终AIMessage。"""
    def __init__(self, delegate, events, kind, gate=None):
        """不同用途共用账本，避免摘要复用于两个方案时重复计算实际账单。"""
        self.delegate, self.events, self.kind, self.gate = delegate, events, kind, gate

    def bind(self, **kwargs):
        """保留同一计量账本和限速器，支持生产提取器自行绑定JSON Mode。"""
        return type(self)(self.delegate.bind(**kwargs), self.events, self.kind, self.gate)

    async def ainvoke(self, messages):
        """保留提供方实际token用量与错误类型；缺失用量用None表示。"""
        event = {"kind": self.kind, "input_tokens": None, "output_tokens": None}
        event["evaluation_queue_seconds"] = await self.gate.wait() if self.gate else 0
        start = time.perf_counter()
        try:
            response = await self.delegate.ainvoke(messages)
            usage = response.usage_metadata or {}
            event.update(input_tokens=usage.get("input_tokens"), output_tokens=usage.get("output_tokens"))
            record_usage(getattr(self.delegate, "model_name", "unknown"), "chat", usage, stage=self.kind)
            return response
        except BaseException as error:
            event["error"] = type(error).__name__
            status = getattr(error, "status_code", None)
            event["status_code"] = status
            if self.gate:
                self.gate.block_access(status)
            record_usage(getattr(self.delegate, "model_name", "unknown"), "chat", stage=self.kind, error=type(error).__name__)
            raise
        finally:
            event["seconds"] = time.perf_counter() - start
            self.events.append(event)


def seed_case(repository, case):
    """逐条通过真实租约/事务归档，并提取规则事实；不写入答案标签，未来数据尚未发生。"""
    from memory.long_term import extract_memories
    from services.chat_service import ChatService
    mapping = {}
    for row in visible_history(case):
        repository.clock = lambda row=row: datetime.fromisoformat(row["created_at"]).timestamp()
        scope = ChatService.session_scope(row["user_id"], row["session_id"])
        assert repository.claim(scope, row["user_id"], row["session_id"], "seed", 3600)
        turn_id, _ = repository.start_request(scope, row["user_id"], "seed", row["id"], row["question"],
                                              extract_memories(row["question"]))
        if row["status"] == "completed":
            repository.finish_request(scope, "seed", turn_id, row["answer"])
        repository.release(scope, "seed")
        mapping[turn_id] = row["id"]
    repository.clock = lambda: datetime.fromisoformat(case["question_time"]).timestamp()
    return mapping


def planned_filters(case):
    """与聊天入口共用问题路由；独立检索测试无近期排除，路由不触发则明确返回空结果。"""
    plan = history_query(case["question"], case["session_id"], datetime.fromisoformat(case["question_time"]).timestamp())
    return plan, dict(session_id=plan.session_id, after=plan.after, before=plan.before) if plan else {}


def retrieval_run(repository, case, mapping, index, settings):
    """固定实现下对BM25、语义、混合分别评分；单独计冷索引构建，比较时均使用已建索引。"""
    plan, filters = planned_filters(case)
    start = time.perf_counter()
    if index:
        candidates = repository.history_turns(case["user_id"], limit=settings["max_turns"], **filters)
        if candidates:
            with usage_scope(stage="history_index_build"):
                index._sync(case["user_id"], candidates)
    result = {"index_build_seconds": time.perf_counter() - start, "methods": {}, "router_triggered": plan is not None}
    methods = {"bm25": HistoryRetriever(repository, **settings)}
    if index:
        methods.update(semantic=SemanticOnly(repository, semantic_index=index, **settings),
                       hybrid=HistoryRetriever(repository, semantic_index=index, **settings))
    names = list(methods)
    offset = int(case["id"].rsplit("-", 1)[-1]) % len(names)
    for name in names[offset:] + names[:offset]:
        retriever = methods[name]
        started = time.perf_counter()
        before_errors = sum("error" in event for event in index.embeddings.events) if index else 0
        with usage_scope(stage="diagnostic_retrieval_" + name):
            turns = retriever.search(case["user_id"], plan.text, **filters) if plan else []
        ids = [mapping.get(row["turn_id"], "UNMAPPED:" + row["turn_id"]) for row in turns]
        degraded = index and sum("error" in event for event in index.embeddings.events) > before_errors
        result["methods"][name] = {"ids": ids, "matched_by": [row["matched_by"] for row in turns],
                                  "seconds": time.perf_counter() - started, "degraded": bool(degraded),
                                  **retrieval_score(case, ids, settings["result_limit"])}
    return result


def full_context(case, cap=32768):
    """完整历史参考仅用本人已完成且已发生的轮次，超预算报错，绝不截断后冒称完整历史。"""
    messages = []
    for row in visible_history(case, own_only=True, completed_only=True):
        messages.extend([{"role": "user", "content": f"[来源会话:{row['session_id']} 时间:{row['created_at']}]\n{row['question']}"},
                         {"role": "assistant", "content": row["answer"]}])
    messages.append({"role": "user", "content": case["question"]})
    if message_tokens(messages) > cap:
        raise ValueError("完整历史参考超过独立预算，不能悄悄截断")
    return messages


async def contexts(repository, case, index, fact_index, summarizer, service_options):
    """调用现有ChatService取得当前方案输入，再按相同近期窗口生成两个消融方案。"""
    from services.chat_service import ChatService
    from memory.conflict_resolution import sanitize_reference
    from memory.session_lease import SessionLeaseStore
    service = ChatService(None, SessionLeaseStore(), repository, summarizer=summarizer,
                          semantic_index=fact_index, history_retriever=HistoryRetriever(repository, semantic_index=index),
                          **service_options)
    start = time.perf_counter()
    turn = await service.open_turn(case["session_id"], case["question"], case["user_id"])
    try:
        snapshot = repository.snapshot(turn.scope, turn.owner, service.history_window)
        all_recent = list(reversed(repository.history(turn.scope, case["user_id"], limit=100)))
        # 开放请求的本轮原文应保留，旧失败请求不可进入消融上下文。
        all_recent = [row for row in all_recent if row.get("status") == "completed" or row["turn_id"] == turn.turn_id]
        recent = all_recent[-service.history_window:]
        after_summary = [row for row in snapshot["recent"] if row["id"] > snapshot["summary_upto"]]
        summary = sanitize_reference(snapshot["summary"], repository.superseded_values(case["user_id"]))
        as_of = datetime.fromisoformat(case["question_time"]).timestamp()
        return {"recent": compose_context(recent, "", [], [], service.context_budget, as_of=as_of),
                "summary_recent": compose_context(after_summary, summary, [], [], service.context_budget, as_of=as_of),
                "memory": turn.messages, "full_history": full_context(case)}, {
                    "seconds": time.perf_counter() - start, "compression": str(turn.compression),
                    "summary": summary, "summary_upto": snapshot["summary_upto"]}
    finally:
        await service.close_turn(turn)


async def answer_one(model, case, messages, events):
    """只按给定上下文调用真实模型，允许一次格式纠正，不用参考答案修补事实错误。"""
    prompt = [{"role": "system", "content": SYSTEM + "\n" + MEMORY_ANSWER_POLICY + f"\n提问时间:{case['question_time']} 当前会话:{case['session_id']}"}] + messages
    started, first = time.perf_counter(), len(events)
    result = {"passed": False, "human_correct": None, "context_estimated_units": message_tokens(messages)}
    for attempt in range(2):
        try:
            response = await asyncio.wait_for(model.ainvoke(prompt), timeout=60)
            text = response.content
            if not isinstance(text, str):
                raise ValueError("回答不是文本")
            result["raw"] = text
            payload = json_payload(text)
            result["format_normalized"] = payload != text.strip()
            parsed = ShortAnswer.model_validate_json(payload)
            result.update(answer=parsed.answer, abstain=parsed.abstain, structure_valid=True, format_repairs=attempt,
                          **answer_score(case, parsed.answer, parsed.abstain))
            break
        except (ValueError, TypeError) as error:
            result.update(error=type(error).__name__, structure_valid=False)
            if attempt == 0:
                prompt += [{"role": "assistant", "content": result.get("raw", "")},
                           {"role": "user", "content": '只纠正格式，输出含answer字符串与abstain布尔值的JSON对象，不要Markdown。'}]
        except Exception as error:
            result.update(error=type(error).__name__, structure_valid=False)
            break
    result.update(wall_seconds=time.perf_counter() - started, calls=events[first:])
    result["seconds"] = sum(event["seconds"] for event in result["calls"])
    if result.get("structure_valid"):
        result.pop("error", None)
    return result


async def evaluate_case(case, directory, embedding, answer_delegate, summary_delegate, config, gate=None):
    """每题独立临时SQL/Chroma与用户，失败照样保留在结果中；不会读取或改写业务库。"""
    from db.memory_repository import MemoryRepository
    repository = MemoryRepository("sqlite:///" + (directory / "memory.db").as_posix())
    repository.initialize()
    index = fact_index = chat_index = None
    embed = CountedEmbeddings(embedding) if embedding else None
    events = []
    result = {"id": case["id"], "category": case["category"], "tags": case["tags"], "answers": {}, "calls": events}
    try:
        mapping = await asyncio.to_thread(seed_case, repository, case)
        history_settings = {key: value for key, value in config["history_retrieval"].items() if key != "semantic"}
        if embed:
            from memory.history_index import HistorySemanticIndex
            from memory.semantic_index import SemanticMemoryIndex
            semantic = {key: value for key, value in config["history_retrieval"]["semantic"].items() if key not in ("enabled", "directory")}
            index = HistorySemanticIndex(repository, directory / "history", embed, **semantic)
            fact_index = SemanticMemoryIndex(repository, directory / "facts", embed,
                                            min_similarity=config["semantic"]["min_similarity"])
        result["retrieval"] = await asyncio.to_thread(retrieval_run, repository, case, mapping, index, history_settings)
        if answer_delegate:
            if embed:
                # 诊断三路检索和当前聊天各自建索引，费用不借用预热，也不把诊断建设量算作生产建设量。
                chat_index = HistorySemanticIndex(repository, directory / "chat-history", embed, **semantic)
            with usage_scope(stage="memory_context"):
                current_contexts, setup = await contexts(repository, case, chat_index, fact_index,
                                    CountedModel(summary_delegate, events, "summary", gate), config["service"])
            result["context_setup"] = setup
            # 同一题四种方案轮换次序；同题摘要只生成一次，两个方案复用，实际账单不重复算。
            offset = int(case["id"].rsplit("-", 1)[-1]) % 4
            for arm in ARMS[offset:] + ARMS[:offset]:
                result["answers"][arm] = await answer_one(CountedModel(answer_delegate, events, arm, gate), case,
                                                          current_contexts[arm], events)
                result["answers"][arm]["context"] = current_contexts[arm]
    except Exception as error:
        result["error"] = type(error).__name__ + ": " + str(error)[:300]
    finally:
        result["embedding"] = embed.events if embed else []
        if fact_index:
            fact_index.close()
        if index:
            index.close()
        if chat_index:
            chat_index.close()
        repository.engine.dispose()
    return result


def fingerprint(paths):
    """记录每个实现/配置文件的SHA256，便于确认验收期间未根据结果调参。"""
    return {str(path.relative_to(ROOT)).replace("\\", "/"): hashlib.sha256(path.read_bytes()).hexdigest() for path in paths}


def usage_total(events):
    """汇总确实有用量的调用并报告覆盖率；缺失或失败调用不能被当成零token。"""
    measured = [event for event in events if event.get("input_tokens") is not None and event.get("output_tokens") is not None]
    return {"calls": len(events), "usage_reported_calls": len(measured),
            "input_tokens_reported": sum(event["input_tokens"] for event in measured) if measured else None,
            "output_tokens_reported": sum(event["output_tokens"] for event in measured) if measured else None,
            "failed_calls": sum("error" in event for event in events)}


def aggregate(cases, rows):
    """所有选中案例都进入分母；检索异常和未生成答案按失败计，并保留分类结果。"""
    by_id = {row["id"]: row for row in rows}
    positive = [case for case in cases if case["gold_turn_ids"]]
    negative = [case for case in cases if not case["gold_turn_ids"]]
    methods = sorted({name for row in rows for name in row.get("retrieval", {}).get("methods", {})})
    scores = {}
    for name in methods:
        def score(case):
            """缺失结果作为未命中保留在固定分母。"""
            record = by_id.get(case["id"], {}).get("retrieval", {}).get("methods", {}).get(name, {})
            return {**record, **retrieval_score(case, record.get("ids", []))}
        scores[name] = {"positive_cases": len(positive), "negative_cases": len(negative),
                        "hit_at_3": sum(score(case).get("hit", False) for case in positive) / len(positive) if positive else None,
                        "recall_at_3": sum(score(case).get("recall", 0) or 0 for case in positive) / len(positive) if positive else None,
                        "mrr_at_3": sum(score(case).get("reciprocal_rank", 0) for case in positive) / len(positive) if positive else None,
                        "all_evidence_at_3": sum(score(case).get("all_evidence", False) for case in positive) / len(positive) if positive else None,
                        "precision_at_3": sum(score(case)["precision_at_k"] for case in positive) / len(positive) if positive else None,
                        "precision_returned_macro": sum(score(case)["precision_returned"] for case in positive) / len(positive) if positive else None,
                        "precision_at_3_label_ceiling": sum(min(3, len(case["gold_turn_ids"])) / 3 for case in positive) / len(positive) if positive else None,
                        "relevant_returned": sum(score(case)["relevant_returned"] for case in positive),
                        "returned_on_positive": sum(score(case)["returned_count"] for case in positive),
                        "negative_nonempty": sum(score(case).get("returned_on_negative", False) for case in negative),
                        "invalid_sources": sum(score(case).get("invalid_sources", 0) for case in cases),
                        "degraded_cases": sum(score(case).get("degraded", False) for case in cases),
                        "latency_p50": percentile([score(case).get("seconds") for case in cases], .5),
                        "latency_p95": percentile([score(case).get("seconds") for case in cases], .95)}
    answers = {}
    if any(row.get("answers") for row in rows):
        for arm in ARMS:
            records = [by_id.get(case["id"], {}).get("answers", {}).get(arm, {}) for case in cases]
            answers[arm] = {"cases": len(cases), "passed": sum(record.get("passed", False) for record in records),
                            "rule_pass_rate": sum(record.get("passed", False) for record in records) / len(cases),
                            "structure_valid": sum(record.get("structure_valid", False) for record in records),
                            "forbidden_answer_cases": sum(bool(record.get("forbidden_matches")) for record in records),
                            "latency_p50": percentile([record.get("seconds") for record in records], .5),
                            "latency_p95": percentile([record.get("seconds") for record in records], .95),
                            "mean_context_estimated_units": sum(record.get("context_estimated_units", 0) for record in records) / len(cases),
                            "usage": usage_total([event for record in records for event in record.get("calls", [])])}
            answers[arm]["abstention"] = abstention_metrics(cases, records)
            no_evidence = [(case, record) for case, record in zip(cases, records) if not case["gold_turn_ids"]]
            answers[arm]["no_evidence_abstention"] = abstention_metrics(
                [pair[0] for pair in no_evidence], [pair[1] for pair in no_evidence])
    costs = combine_reports([row["usage_ledger"] for row in rows if "usage_ledger" in row])
    return {"cases": len(cases), "recorded_cases": len(rows), "errors": sum("error" in row for row in rows),
            "retrieval": scores, "answers": answers,
            "actual_llm_usage": usage_total([event for row in rows for event in row.get("calls", [])]),
            "summary_usage_shared_once": usage_total([event for row in rows for event in row.get("calls", []) if event["kind"] == "summary"]),
            "embedding_sdk_operations": sum(len(row.get("embedding", [])) for row in rows),
            "embedding_input_bytes": sum(event["input_bytes"] for row in rows for event in row.get("embedding", [])),
            "embedding_token_usage": costs["embedding_input_tokens_reported"] if costs["cases"] else None,
            "reference_costs": costs if costs["cases"] else None}


async def run(args):
    """冻结数据与实现指纹，写增量结果并限制并行；真实调用仅作用于本进程和合成数据。"""
    import yaml
    if args.env_file:
        from dotenv import load_dotenv
        load_dotenv(args.env_file, override=True)
    os.environ["LANGFUSE_PUBLIC_KEY"] = ""
    os.environ["LANGFUSE_SECRET_KEY"] = ""
    dataset = json.loads(args.dataset.read_text(encoding="utf-8"))
    cases = [case for case in validate_dataset(dataset) if case["split"] == args.split]
    if args.limit:
        cases = cases[:args.limit]
    config = yaml.safe_load((ROOT / "config/memory.yml").read_text(encoding="utf-8"))
    sources = [ROOT / path for path in ("memory/history_retrieval.py", "memory/history_index.py", "services/chat_service.py",
               "memory/context_builder.py", "memory/long_term.py", "memory/fact_rules.py", "memory/answer_policy.py", "memory/batch_summary.py",
               "db/memory_repository.py", "db/memory_schedule.py", "db/memory_extensions.py", "model/factory.py", "model/usage.py",
               "services/json_payload.py", "config/model_prices.yml", "config/memory.yml", "config/rag.yml")]
    manifest = {"dataset_sha256": hashlib.sha256(args.dataset.read_bytes()).hexdigest(), "implementation_sha256": fingerprint(sources),
                "answer_system_prompt": SYSTEM + "\n" + MEMORY_ANSWER_POLICY,
                "evaluator_sha256": fingerprint([Path(__file__), ROOT / "evals/history_metrics.py"]),
                "split": args.split, "case_ids": [case["id"] for case in cases], "min_call_interval": args.min_call_interval,
                "mode": "real_models" if args.live else "real_embedding_retrieval" if args.embedding else "offline_bm25",
                "dataset_scope": dataset["scope"],
                "limits": ["合成集含同类模板，未人工二次审核，不是外部盲测或线上表现",
                           "SQLite临时库、规则事实提取；没有测试生产MySQL、Redis命中或模型事实提取",
                           "未来消息模拟尚未发生，不入库；不能据此证明数据库已存在未来行时的防泄漏",
                           "四方案均直接回答历史问题，不运行产品知识库/业务工具Agent；完整历史是参考上限而非Oracle",
                           "模型短答案规则评分，human_correct仍待人工复核；上下文估算单位不是实际token",
                           "摘要一次生成两个方案复用；摘要和索引建设成本单列，Embedding按实际usage计量，缺失不能当0",
                           "Embedding记录SDK操作数，一次操作可包含多批HTTP请求及重试，不等于网络请求数"]}
    args.output.mkdir(parents=True, exist_ok=True)
    manifest_path = args.output / "manifest.json"
    if manifest_path.exists():
        raise ValueError("输出目录已有manifest，请使用新目录，避免混合不同运行")
    manifest_path.write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
    embedding = answer_delegate = summary_delegate = None
    if args.embedding or args.live:
        from model.factory import chat_model, embed_model
        embedding = embed_model
        manifest["embedding_model"] = embed_model.model
        if args.live:
            # 官方Omni仅支持流式，必须设置模型对象本身的streaming属性，让ainvoke聚合流。
            chosen = args.chat_model or chat_model.model_name
            configured = chat_model.model_copy(update={"model_name": chosen, "streaming": True, "stream_usage": True, "temperature": 0, "max_tokens": 512, "max_retries": 1})
            answer_delegate = configured.bind(extra_body={"enable_thinking": False}) if chosen.startswith("qwen") else configured
            summary_delegate = answer_delegate
            manifest["chat_model"] = chosen
    manifest_path.write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
    rows, semaphore, gate = [], asyncio.Semaphore(args.concurrency), RequestGate(args.min_call_interval)
    async def one(case):
        """每题只写一行终态，持续输出进度；单题失败不会丢掉其他已完成结果。"""
        async with semaphore:
            if gate.blocked_status:
                row = {"id": case["id"], "category": case["category"], "tags": case["tags"], "answers": {}, "calls": [],
                       "error": "not_run_provider_access_" + str(gate.blocked_status), "status": "not_run"}
                row["usage_ledger"] = UsageLedger().report()
                rows.append(row)
                with (args.output / "results.jsonl").open("a", encoding="utf-8") as stream:
                    stream.write(json.dumps(row, ensure_ascii=False) + "\n")
                print(f"[{len(rows)}/{len(cases)}] {case['id']} skipped: provider access blocked", flush=True)
                return
            with tempfile.TemporaryDirectory(prefix="history-eval-") as directory:
                ledger = UsageLedger()
                with usage_scope(ledger, "setup"):
                    row = await evaluate_case(case, Path(directory), embedding, answer_delegate, summary_delegate, config, gate)
                row["usage_ledger"] = ledger.report()
            rows.append(row)
            with (args.output / "results.jsonl").open("a", encoding="utf-8") as stream:
                stream.write(json.dumps(row, ensure_ascii=False) + "\n")
            status = row.get("error", " ".join(f"{arm}={int(data['passed'])}" for arm, data in row["answers"].items()) or "retrieval_done")
            print(f"[{len(rows)}/{len(cases)}] {case['id']} {status}", flush=True)
    await asyncio.gather(*(one(case) for case in cases))
    if fingerprint(sources) != manifest["implementation_sha256"]:
        raise RuntimeError("验收期间实现发生变化，本轮报告不可用于固定实现比较")
    summary = {"manifest": manifest, "provider_access_blocked": gate.blocked_status, "overall": aggregate(cases, rows),
               "categories": {category: aggregate([case for case in cases if case["category"] == category],
                             [row for row in rows if row["category"] == category]) for category in sorted({case["category"] for case in cases})}}
    (args.output / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(summary["overall"], ensure_ascii=False, indent=2), flush=True)


def main():
    """默认不联网；--embedding测真实向量，--live另测真实摘要及四种上下文回答。"""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", type=Path, default=ROOT / "evals/history_cases.json")
    parser.add_argument("--split", choices=("calibration", "test"), default="test")
    parser.add_argument("--limit", type=int)
    parser.add_argument("--embedding", action="store_true")
    parser.add_argument("--live", action="store_true")
    parser.add_argument("--chat-model", help="只覆盖本次评测的模型名，不写config/rag.yml；不同模型结果须单独报告")
    parser.add_argument("--env-file", type=Path)
    parser.add_argument("--concurrency", type=int, default=2)
    parser.add_argument("--min-call-interval", type=float, default=1.25,
                        help="评测模型请求的最短秒间隔，默认1.25秒；排队时间单列，不计入模型延迟")
    parser.add_argument("--output", type=Path, default=ROOT / "evals/runs" / ("history-" + datetime.now().strftime("%Y%m%d-%H%M%S")))
    args = parser.parse_args()
    if args.concurrency < 1 or args.min_call_interval < 0 or (args.limit is not None and args.limit < 1):
        parser.error("concurrency和limit必须为正数")
    asyncio.run(run(args))


if __name__ == "__main__":
    main()
