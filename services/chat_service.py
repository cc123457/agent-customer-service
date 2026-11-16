from agent.react_agent import ReactAgent
from memory.session_store import SessionStore


class ChatService:
    """对话业务逻辑：管理会话记忆（含压缩记忆），驱动 Agent 流式生成回答。

    这一层不接触任何 HTTP 细节（不 import fastapi），只处理「纯业务」，
    因此可以直接被单元测试、也可以被其他入口（如消息机器人）复用。
    """

    def __init__(self, agent: ReactAgent, store: SessionStore, summarizer=None, keep_after_compress: int = None):
        self.agent = agent
        self.store = store
        self._summarizer = summarizer   # 摘要用的 LLM；None 表示首次用时再懒加载
        # 压缩后保留的消息条数：默认压完留一半；并钳制到 [1, max_history-1]，
        # 否则 keep 为负会清空历史、keep 过大则永不触发压缩
        if keep_after_compress is None:
            keep = max(1, store.max_history // 2)
        else:
            keep = max(1, min(keep_after_compress, store.max_history - 1))
        self.keep_after_compress = keep

    # ---- 记忆 ----

    def build_messages(self, session_id: str, user_input: str) -> list[dict]:
        """把本轮用户输入存入会话，返回完整上下文（摘要 + 历史）。

        历史达到水位（max_history）时，批量把最早的一部分压成摘要，
        而不是逐条截断 —— 这就是「压缩记忆（水位触发 + 批量）」。
        """
        self.store.append(session_id, "user", user_input)
        self._maybe_compress(session_id)

        history = self.store.get(session_id)
        summary = self.store.get_summary(session_id)
        if summary:
            # 摘要作为一条带标注的 user 消息前置，避免与 Agent 的 system_prompt 冲突
            return [{"role": "user", "content": f"[对话历史摘要，回答时请结合] {summary}"}] + history
        return history

    def save_reply(self, session_id: str, reply: str):
        """流结束后，把完整回答写回会话。不在此处压缩——统一由下一轮 build_messages 触发。"""
        self.store.append(session_id, "assistant", reply)

    # ---- 流式 ----

    def stream_reply(self, messages: list[dict], user_id: str | None = None):
        """透传 Agent 的异步流式输出（async generator，每段为 {type, content}）。

        注意：MCP 工具仅支持异步调用，所以整条链路必须走异步。
        """
        return self.agent.execute_stream(messages, user_id=user_id)

    # ---- 压缩记忆（摘要） ----

    def _maybe_compress(self, session_id: str):
        """水位触发：历史达到 max_history 时，一次性把最早的一部分批量压成摘要。

        不是「每溢出 1 条压 1 次」，而是「攒满一个水位，压掉一批」——
        压缩频率比逐条压缩降低一个数量级，且每次摘要的信息量更完整。
        """
        history = self.store.get(session_id)
        if len(history) < self.store.max_history:
            return
        to_compress = len(history) - self.keep_after_compress
        if to_compress <= 0:
            return
        oldest = self.store.extract_oldest(session_id, to_compress)
        self._update_summary(session_id, oldest)

    def _update_summary(self, session_id: str, messages_to_summarize: list):
        """用 LLM 增量合并摘要；失败则静默降级（保留旧摘要，被取出的消息直接丢弃）。"""
        if not messages_to_summarize:
            return
        try:
            old = self.store.get_summary(session_id) or ""
            text = "\n".join(f"{m['role']}: {m['content']}" for m in messages_to_summarize)
            prompt = (
                "你是对话摘要助手。请把新的对话片段合并进旧摘要，输出更新后的完整摘要。"
                "保留关键信息（用户关心的问题、已经给出的重要结论或数据），保持简洁、不要重复。\n\n"
                f"旧摘要：\n{old}\n\n"
                f"新对话片段：\n{text}"
            )
            resp = self._get_summarizer().invoke(prompt)
            new_summary = resp.content.strip() if hasattr(resp, "content") else str(resp).strip()
            self.store.set_summary(session_id, new_summary)
        except Exception:
            # 摘要失败不拖垮主链路：退化为「截断记忆」
            pass

    def _get_summarizer(self):
        if self._summarizer is None:
            from model.factory import chat_model
            self._summarizer = chat_model
        return self._summarizer
