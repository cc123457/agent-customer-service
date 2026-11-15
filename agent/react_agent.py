import asyncio
import os

from langchain.agents import create_agent
from langchain_mcp_adapters.client import MultiServerMCPClient

from model.factory import chat_model
from utils.prompt_loader import load_system_prompts
from utils.observability import get_langfuse_handler
from agent.tools.agent_tools import (rag_summarize, get_user_location, get_user_id,
                                     get_current_month, fetch_external_data, fill_context_for_report)
from agent.tools.middleware import monitor_tool, log_before_model, report_prompt_switch


class ReactAgent:
    def __init__(self):
        # 内置工具（自己写的 @tool）；天气工具改由 MCP 服务提供
        builtin_tools = [rag_summarize, get_user_location, get_user_id,
                         get_current_month, fetch_external_data, fill_context_for_report]
        mcp_tools = self._load_weather_mcp_tools()

        self.agent = create_agent(
            model=chat_model,
            system_prompt=load_system_prompts(),
            tools=builtin_tools + mcp_tools,
            middleware=[monitor_tool, log_before_model, report_prompt_switch],
        )
        # Langfuse 链路追踪（可选：未配置 key 则为 None，不启用追踪）
        self.langfuse_handler = get_langfuse_handler()

    def _load_weather_mcp_tools(self):
        """从天气 MCP 服务加载工具；服务不可用时降级为空列表，不影响其他能力"""
        url = os.getenv("WEATHER_MCP_URL", "http://127.0.0.1:8001/mcp")
        try:
            return asyncio.run(self._load_weather_mcp_tools_async(url))
        except Exception:
            return []

    async def _load_weather_mcp_tools_async(self, url: str):
        client = MultiServerMCPClient({
            "weather": {"transport": "streamable_http", "url": url},
        })
        return await client.get_tools()

    async def execute_stream(self, messages: list[dict], user_id: str | None = None):
        """接收完整的对话历史（含本轮用户消息），异步流式返回结构化输出。

        每段输出是一个 dict：
        - {"type": "action", "content": "调用工具 xxx，参数 ..."}  # Agent 动作轨迹（思考可见化）
        - {"type": "answer", "content": "..."}                      # 最终回答

        user_id：当前登录用户，通过 runtime context 传给工具（供 get_user_id 使用）。

        注意：必须走异步链路——MCP 工具仅支持异步调用，同步调用会 NotImplementedError。
        """
        input_dict = {
            "messages": messages,
        }

        # runtime context：report 用于提示词切换；user_id 用于工具获取当前用户
        context = {"report": False}
        if user_id:
            context["user_id"] = user_id

        # 第三个参数 context 是 runtime 上下文
        # 若启用了 Langfuse，则挂上 callback handler，自动追踪本次执行的完整链路
        stream_kwargs = {"stream_mode": "values", "context": context}
        if self.langfuse_handler:
            stream_kwargs["config"] = {
                "callbacks": [self.langfuse_handler],
                "metadata": {"langfuse_user_id": user_id or "anonymous"},
            }

        # 异步流式（MCP 工具仅支持异步调用）
        async for chunk in self.agent.astream(input_dict, **stream_kwargs):
            latest_message = chunk["messages"][-1]

            # 1. 模型发起工具调用 → 作为「动作」输出（把 ReAct 的思考动作化）
            if latest_message.type == "ai" and latest_message.tool_calls:
                for tc in latest_message.tool_calls:
                    name = tc.get("name", "unknown")
                    args = tc.get("args", {})
                    yield {"type": "action", "content": f"调用工具 {name}，参数 {args}"}

            # 2. 最终回答（AI 消息、无工具调用、content 非空）
            elif latest_message.type == "ai" and latest_message.content:
                yield {"type": "answer", "content": latest_message.content.strip()}


if __name__ == '__main__':
    async def _main():
        agent = ReactAgent()

        async for item in agent.execute_stream([{"role": "user", "content": "给我生成我的使用报告"}]):
            print(f"[{item['type']}] {item['content']}", flush=True)

    asyncio.run(_main())
