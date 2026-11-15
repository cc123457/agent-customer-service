from agent.react_agent import ReactAgent
from memory.session_store import SessionStore


class ChatService:
    """对话业务逻辑：管理会话记忆，驱动 Agent 流式生成回答。

    这一层不接触任何 HTTP 细节（不 import fastapi），只处理「纯业务」，
    因此可以直接被单元测试、也可以被其他入口（如消息机器人）复用。
    """

    def __init__(self, agent: ReactAgent, store: SessionStore):
        self.agent = agent
        self.store = store

    def build_messages(self, session_id: str, user_input: str) -> list[dict]:
        """把本轮用户输入存入会话，返回完整对话历史"""
        self.store.append(session_id, "user", user_input)
        return self.store.get(session_id)

    def stream_reply(self, messages: list[dict], user_id: str | None = None):
        """透传 Agent 的异步流式输出（async generator，每段为 {type, content}）。

        注意：MCP 工具仅支持异步调用，所以整条链路必须走异步。
        """
        return self.agent.execute_stream(messages, user_id=user_id)

    def save_reply(self, session_id: str, reply: str):
        """流结束后，把完整回答写回会话"""
        self.store.append(session_id, "assistant", reply)
