"""聊天业务唯一入口：租约、原文归档、摘要/召回、Agent执行及回答保存与恢复。"""
import asyncio
import logging
import time
import re
from uuid import uuid4
from urllib.parse import quote

from memory.batch_summary import BatchSummarizer
from memory.budget import estimate_tokens, message_tokens
from memory.context_builder import compose_context
from memory.long_term import extract_memories, recall_memories
from memory.history_retrieval import HistoryRetriever, history_query
from memory.history_retrieval import parse_history_scope
from memory.request_analyzer import MemoryRequestAnalyzer
from services.chat_types import ChatTurn, CompressionStatus, ContextBudgetError, SessionBusyError

logger = logging.getLogger(__name__)


class ChatService:
    """组织当前聊天生命周期；HTTP、SQL、租约、摘要和检索分别留在对应模块。"""

    def __init__(self, agent, leases, repository, summarizer=None, history_window=20,
                 keep_after_compress=None, context_budget=8192, summary_budget=1024,
                 summary_input_budget=16000, summary_timeout=20, retry_cooldown=30,
                 turn_timeout=90, lease_seconds=120, clock=time.monotonic,
                 max_summary_batches=8, compression_timeout=40, long_term_budget=1200,
                 archive_budget=1600, request_analyzer=None, semantic_index=None, history_retriever=None):
        """组装唯一聊天流程并校验配置；SQL保存历史，leases只管互斥，模型在请求执行时才调用。"""
        if history_window < 2:
            raise ValueError("近期窗口必须至少保留两条消息")
        if not 0 < summary_budget < context_budget:
            raise ValueError("摘要预算必须大于 0 且小于会话上下文预算")
        if summary_input_budget <= summary_budget or summary_timeout <= 0 or retry_cooldown < 0:
            raise ValueError("摘要输入预算、超时或重试间隔无效")
        if turn_timeout <= 0 or lease_seconds <= turn_timeout + 5:
            raise ValueError("会话租约必须大于请求时间上限并留出存储操作余量")
        self.agent = agent
        self.leases = leases
        self._summarizer = summarizer
        keep = history_window // 2 if keep_after_compress is None else keep_after_compress
        self.keep_after_compress = max(1, min(keep, history_window - 1))
        self.context_budget = context_budget
        self.summary_budget = summary_budget
        self.summary_input_budget = summary_input_budget
        self.summary_timeout = summary_timeout
        self.retry_cooldown = retry_cooldown
        self.turn_timeout = turn_timeout
        self.lease_seconds = lease_seconds
        self._clock = clock
        self._retry_after = {}
        if max_summary_batches < 1 or compression_timeout <= 0 or long_term_budget < 0 or archive_budget < 0:
            raise ValueError("分批摘要与长期记忆预算配置无效")
        self.history_window = history_window
        self.repository = repository
        self.max_summary_batches = max_summary_batches
        self.compression_timeout = compression_timeout
        self.long_term_budget = long_term_budget
        self.archive_budget = archive_budget
        self.request_analyzer = request_analyzer or MemoryRequestAnalyzer()
        self.semantic_index = semantic_index
        self.history_retriever = history_retriever if history_retriever is not None else HistoryRetriever(repository)

    @staticmethod
    def session_scope(user_id: str, session_id: str) -> str:
        """组合后端认证的用户 ID 与客户端会话 ID，返回转义后的存储 scope。

        open_turn 调用本方法，使相同 session_id 在不同用户下对应不同数据。
        这里只组合标识，不验证 JWT；身份认证由 api/deps.py 提前完成。
        """
        if not user_id or not user_id.strip() or not session_id or not session_id.strip():
            raise ValueError("用户身份和会话 ID 不能为空")
        return f"{quote(user_id, safe='')}:{quote(session_id, safe='')}"

    async def open_turn(self, session_id, user_input, user_id, request_id=None):
        """接口调用：获取双层租约、事务归档用户消息与明确事实，再组装有限模型上下文。

        客户端可传request_id去重；超时/准备失败标记failed、释放租约，原文保留。
        Redis或内存租约是快速互斥，数据库租约仍保护Redis过期/丢失后的权威写入。
        """
        if not user_input.strip():
            raise ValueError("用户输入不能为空")
        if message_tokens([{"role": "user", "content": user_input}]) > self.context_budget:
            raise ContextBudgetError("本轮消息超过上下文预算，请缩短输入")
        scope = self.session_scope(user_id, session_id)
        owner = uuid4().hex
        request_id = request_id or uuid4().hex
        deadline = asyncio.get_running_loop().time() + self.turn_timeout
        claimed = await asyncio.to_thread(self.leases.claim, scope, owner, self.lease_seconds)
        if not claimed:
            raise SessionBusyError("该会话正在处理上一条消息")
        turn_id = None
        try:
            async with asyncio.timeout_at(deadline):
                claimed = await asyncio.to_thread(
                    self.repository.claim, scope, user_id, session_id, owner, self.lease_seconds,
                )
                if not claimed:
                    raise SessionBusyError("数据库中该会话仍有未结束请求")
                before = await asyncio.to_thread(self.repository.snapshot, scope, owner, 4)
                before_policy = await asyncio.to_thread(self.repository.access_policy, user_id)
                # 分析器也只能看安全的小窗口；不把已经撤回的原文交给路由模型。
                previous = before_policy.messages(before["recent"])
                known = await asyncio.to_thread(self.repository.deletion_targets, user_id) if re.search(r"忘记|删除|别再记|不再保存", user_input) else []
                before_summary = before["summary"] if before["forget_version"] == before_policy.version else ""
                analysis = await self.request_analyzer.analyze(user_input, previous, before_summary, known)
                turn_id, source_id = await asyncio.to_thread(
                    self.repository.start_request, scope, user_id, owner, request_id, user_input, analysis.operations,
                    before_policy.version,
                )
                snapshot = await asyncio.to_thread(
                    self.repository.snapshot, scope, owner, self.history_window,
                )
                policy = await asyncio.to_thread(self.repository.access_policy, user_id)
                if snapshot["forget_version"] != policy.version:
                    snapshot["summary"] = ""
                snapshot["pending"] = policy.messages(snapshot["pending"])
                snapshot["recent"] = policy.messages(snapshot["recent"])
                status = await self._compress_snapshot(scope, owner, snapshot)
                snapshot = await asyncio.to_thread(
                    self.repository.snapshot, scope, owner, self.history_window,
                )
                policy = await asyncio.to_thread(self.repository.access_policy, user_id)
                if snapshot["forget_version"] != policy.version:
                    snapshot["summary"] = ""
                recent = policy.messages([row for row in snapshot["recent"] if row["id"] > snapshot["summary_upto"]])
                # 当前删除请求不能被其自己新立的屏障移除；模糊删除不携带任何旧上下文。
                current = [row for row in snapshot["recent"] if row["id"] == source_id]
                recent = [row for row in recent if row["id"] != source_id] + current
                if analysis.clarification:
                    recent = current
                if self.semantic_index and analysis.read.search_notes and not analysis.clarification:
                    try:
                        await asyncio.to_thread(self.semantic_index.sync_user, user_id)
                    except Exception:
                        logger.warning("个人记忆向量索引同步失败，保留SQL事实并使用词法检索", exc_info=True)
                facts = (await asyncio.to_thread(self._recall, user_id, user_input, recent, analysis.read)
                         if not analysis.clarification else [])
                facts = policy.facts(facts)
                archive = []
                plan = None
                if analysis.read.search_history and not analysis.clarification:
                    try:
                        scope_plan = parse_history_scope(user_input, session_id, self.repository.clock())
                        from dataclasses import replace
                        plan = replace(scope_plan, text=analysis.read.query or user_input)
                    except ValueError as error:
                        analysis.clarification = str(error)
                if plan is not None:
                    archive = await asyncio.to_thread(
                        self.history_retriever.search, user_id, plan.text,
                        session_id=plan.session_id, after=plan.after, before=plan.before,
                        exclude_turn_ids=list({row["turn_id"] for row in recent}),
                    )
                from memory.conflict_resolution import sanitize_reference
                old_values = await asyncio.to_thread(self.repository.superseded_values, user_id)
                safe_summary = sanitize_reference(snapshot["summary"], old_values) if not analysis.clarification else ""
                # 明确回忆旧事时保留当时原话与时间；普通问题仍遮蔽已更正/遗忘的旧值。
                if plan is None or not plan.preserve_original:
                    archive = [dict(turn, messages=[dict(row, content=sanitize_reference(row["content"], old_values))
                                                   for row in turn["messages"]]) for turn in archive]
                archive = policy.turns(archive)
                if any(row.get("needs_clarification") for row in facts):
                    analysis.clarification = "请先说明是哪台设备或确认当前型号，已有记录存在多个对象或互相矛盾的信息。"
                context = compose_context(
                    recent, safe_summary, facts, archive, self.context_budget,
                    self.long_term_budget, self.archive_budget, as_of=self.repository.clock(),
                )
                await asyncio.to_thread(self.repository.assert_request_current, user_id, turn_id)
                direct = analysis.clarification
                if analysis.forget_requested and not direct and not analysis.read.search_history:
                    direct = "已停止使用指定的个人记忆；原始会话档案仍按保留策略保存。"
            return ChatTurn(
                scope, owner, context, deadline, status, user_id, session_id,
                request_id, turn_id, source_id, direct_answer=direct,
                memory_analysis={"model_called": analysis.model_called, "degraded": analysis.degraded,
                                 "search_history": analysis.read.search_history, "search_notes": analysis.read.search_notes},
            )
        except BaseException:
            if turn_id:
                await asyncio.to_thread(self.repository.fail_request, turn_id)
            try:
                await asyncio.to_thread(self.repository.release, scope, owner)
            finally:
                await asyncio.to_thread(self.leases.release, scope, owner)
            raise

    async def _compress_snapshot(self, scope, owner, snapshot):
        """选择待摘要前缀，逐批合并，全部成功才提交摘要/水位；失败后完整冷却，原文不删除。"""
        pending = snapshot["pending"]
        summary_size = (estimate_tokens("[对话历史摘要，回答时请结合] " + snapshot["summary"]) + 12
                        if snapshot["summary"] else 0)
        over_budget = message_tokens(pending) + summary_size > self.context_budget
        if len(pending) < self.history_window and not over_budget:
            return CompressionStatus.SKIPPED
        now = self._clock()
        self._retry_after = {key: end for key, end in self._retry_after.items() if end > now}
        if self._retry_after.get(scope, 0) > now:
            return CompressionStatus.COOLDOWN
        recent = snapshot["recent"]
        keep = min(len(recent), self.keep_after_compress)
        cutoff = recent[-keep]["id"]
        retained_turns = {row["turn_id"] for row in recent[-keep:]}
        originals = [row for row in pending if row["id"] < cutoff and row["turn_id"] not in retained_turns]
        if not originals and over_budget:
            originals = [row for row in pending if row["id"] < recent[-1]["id"]]
        # 分页边界或近期窗口都可能落在一轮中间；摘要水位只推进到完整问答的末尾。
        if originals and originals[-1]["role"] == "user":
            last_turn = originals[-1]["turn_id"]
            originals = [row for row in originals if row["turn_id"] != last_turn]
        if not originals:
            return CompressionStatus.SKIPPED
        try:
            summarizer = BatchSummarizer(
                self._get_summarizer(), self.summary_input_budget, self.summary_budget,
                self.summary_timeout, self.max_summary_batches,
            )
            summary, _ = await asyncio.wait_for(
                summarizer.summarize(snapshot["summary"], originals), self.compression_timeout,
            )
            committed = await asyncio.to_thread(
                self.repository.commit_summary, scope, owner, snapshot["revision"],
                originals[-1]["id"], summary,
            )
            return CompressionStatus.SUCCESS if committed else CompressionStatus.CONFLICT
        except Exception:
            logger.warning("分批摘要失败，归档原文和旧水位保留：%s", scope, exc_info=True)
            self._retry_after[scope] = self._clock() + self.retry_cooldown
            return CompressionStatus.FAILED

    async def reply_turn(self, turn, user_id):
        """将有限上下文交给Agent，正常结束事务保存回答；失败/取消仅标记请求，不伪造回复。"""
        parts = []
        try:
            if user_id != turn.user_id:
                raise PermissionError("本轮身份与认证用户不一致")
            async with asyncio.timeout_at(turn.deadline):
                await asyncio.to_thread(self.repository.assert_request_current, user_id, turn.turn_id)
                kwargs = {"execution_id": turn.turn_id, "resume": turn.resume} if getattr(self.agent, "checkpointer", None) else {}
                async def items():
                    """明确澄清/遗忘确认由服务给出，避免模型自行挑设备或声称物理清空。"""
                    if turn.direct_answer:
                        yield {"type": "answer", "content": turn.direct_answer}
                    else:
                        async for item in self.agent.execute_stream(turn.messages, user_id=user_id, **kwargs):
                            yield item
                async for item in items():
                    await asyncio.to_thread(self.repository.assert_request_current, user_id, turn.turn_id)
                    if item["type"] == "answer":
                        parts.append(item["content"])
                        if sum(len(part.encode("utf-8")) for part in parts) > self.repository.limits["max_answer_bytes"]:
                            raise ContextBudgetError("回答超过本轮归档长度上限，未保存为成功回答")
                    yield item
                answer = "".join(parts)
                if not answer.strip():
                    raise ValueError("模型未返回有效回答")
                await asyncio.to_thread(
                    self.repository.finish_request, turn.scope, turn.owner, turn.turn_id, answer,
                )
        except BaseException:
            await asyncio.to_thread(self.repository.fail_request, turn.turn_id)
            raise
        finally:
            await self.close_turn(turn)

    async def resume_turn(self, session_id, request_id, user_id):
        """申请新租约后继续失败请求；不追加用户原文，不重做图外记忆写入，重新注入认证身份。"""
        if not getattr(self.agent, "checkpointer", None):
            raise ValueError("当前Agent未配置持久检查点")
        scope = self.session_scope(user_id, session_id)
        row = await asyncio.to_thread(self.repository.request, scope, user_id, request_id)
        if row is None:
            raise LookupError("未找到本人请求")
        config = {"configurable": {"thread_id": row["id"]}}
        if await self.agent.checkpointer.aget_tuple(config) is None:
            raise LookupError("该请求尚无可恢复检查点")
        owner = uuid4().hex
        if not await asyncio.to_thread(self.leases.claim, scope, owner, self.lease_seconds):
            raise SessionBusyError("该会话正在处理上一条消息")
        try:
            if not await asyncio.to_thread(self.repository.claim, scope, user_id, session_id, owner, self.lease_seconds):
                raise SessionBusyError("数据库中该会话仍有未结束请求")
            turn_id, source_id = await asyncio.to_thread(self.repository.resume_request, scope, user_id, owner, request_id)
            deadline = asyncio.get_running_loop().time() + self.turn_timeout
            return ChatTurn(scope, owner, [], deadline, CompressionStatus.SKIPPED,
                                  user_id, session_id, request_id, turn_id, source_id, True)
        except BaseException:
            try:
                await asyncio.to_thread(self.repository.release, scope, owner)
            finally:
                await asyncio.to_thread(self.leases.release, scope, owner)
            raise

    async def list_checkpoints(self, session_id, request_id, user_id):
        """由认证接口调用，只能查看本人请求的检查点概况，不能指定其他人的内部执行ID。"""
        row = await asyncio.to_thread(self.repository.request, self.session_scope(user_id, session_id), user_id, request_id)
        if row is None:
            raise LookupError("未找到本人请求")
        return await self.agent.checkpoints(row["id"])

    def _recall(self, user_id, query, recent, read_plan=None):
        """向量服务失败时退回本人SQL词法检索，不将网络异常伪装成事实不存在。"""
        try:
            return recall_memories(self.repository, user_id, query, recent,
                                   self.semantic_index if read_plan is None or read_plan.search_notes else None, read_plan)
        except Exception:
            logger.warning("个人记忆语义检索不可用，使用SQL词法召回", exc_info=True)
            return recall_memories(self.repository, user_id, query, recent, read_plan=read_plan)

    async def close_turn(self, turn):
        """同时释放权威数据库和短期存储的本轮租约；重复调用只影响匹配owner。"""
        try:
            await asyncio.to_thread(self.repository.release, turn.scope, turn.owner)
        finally:
            await asyncio.to_thread(self.leases.release, turn.scope, turn.owner)

    def _get_summarizer(self):
        """返回摘要模型：优先使用注入对象，否则首次加载全局 chat_model。

        由 _compress_snapshot 调用；此处只获取模型对象，不执行摘要推理。
        """
        if self._summarizer is None:
            from model.factory import chat_model
            self._summarizer = chat_model
        return self._summarizer
