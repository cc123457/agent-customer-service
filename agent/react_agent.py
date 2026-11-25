import asyncio
import os

from langchain.agents import create_agent
from langchain_mcp_adapters.client import MultiServerMCPClient

from model.factory import chat_model
from utils.prompt_loader import load_system_prompts
from utils.observability import get_langfuse_handler
from memory.answer_policy import MEMORY_ANSWER_POLICY
from agent.tools.agent_tools import (rag_summarize, get_user_location, get_user_id,
                                     get_current_month, fetch_external_data, fill_context_for_report)
from agent.tools.middleware import monitor_tool, log_before_model, report_prompt_switch


class ReactAgent:
    def __init__(self, checkpointer=None, model=None, tools=None, system_prompt=None):
        """由 api/deps.py 首次创建：注册内置/MCP工具、中间件，并构造 Agent 图。

        可注入持久化Checkpointer与测试模型；执行线程按请求隔离，跨轮上下文仍由业务层选择。
        """
        # 内置工具（自己写的 @tool）；天气工具改由 MCP 服务提供
        builtin_tools = [rag_summarize, get_user_location, get_user_id,
                         get_current_month, fetch_external_data, fill_context_for_report]
        mcp_tools = self._load_weather_mcp_tools() if tools is None else []
        self.checkpointer = checkpointer

        self.agent = create_agent(
            model=model or chat_model,
            system_prompt=(system_prompt if system_prompt is not None else load_system_prompts()) + "\n" + MEMORY_ANSWER_POLICY,
            tools=tools if tools is not None else builtin_tools + mcp_tools,
            middleware=[monitor_tool, log_before_model, report_prompt_switch],
            checkpointer=checkpointer,
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
        """接收天气 MCP 地址，返回服务公开的工具列表，由同步加载包装方法调用。

        使用 Streamable HTTP 异步发现工具；连接失败向外抛出，再由包装层降级为空列表。
        """
        client = MultiServerMCPClient({
            "weather": {"transport": "streamable_http", "url": url},
        })
        return await client.get_tools()

    async def execute_stream(self, messages: list[dict], user_id: str | None = None, callbacks=None,
                             execution_id=None, resume=False):
        """由 ChatService.reply_turn 调用，接收上下文消息、认证身份和可选回调。

        创建本轮 Runtime Context，执行 Agent 图并将状态更新转换成事件字典；
        图框架负责真正执行工具，本方法只提取调用请求与回答用于展示。
        stream_mode='values' 是状态事件，不是逐 Token 流；action 不表示工具已成功。

        每段输出是一个 dict：
        - {"type": "action", "content": "调用工具 xxx，参数 ..."}  # 工具调用请求展示
        - {"type": "answer", "content": "..."}                      # 最终回答

        user_id：当前登录用户，通过 runtime context 传给工具（供 get_user_id 使用）。

        注意：必须走异步链路——MCP 工具仅支持异步调用，同步调用会 NotImplementedError。
        """
        input_dict = None if resume else {
            "messages": messages,
        }

        # runtime context：report 用于提示词切换；user_id 用于工具获取当前用户
        context = {"report": False}
        if user_id:
            context["user_id"] = user_id

        # 第三个参数 context 是 runtime 上下文
        # 若启用了 Langfuse，则挂上 callback handler，自动追踪本次执行的完整链路
        stream_kwargs = {"stream_mode": "values", "context": context}
        handlers = list(callbacks or [])
        if self.langfuse_handler:
            handlers.append(self.langfuse_handler)
        stream_kwargs["config"] = {
            "callbacks": handlers,
            "metadata": {"langfuse_user_id": user_id or "anonymous"},
            "recursion_limit": 30,
        }
        if self.checkpointer:
            if not execution_id:
                raise ValueError("持久执行必须提供服务端生成的execution_id")
            stream_kwargs["config"]["configurable"] = {"thread_id": execution_id}
            stream_kwargs["durability"] = "sync"
        elif resume:
            raise ValueError("未配置持久检查点，不能恢复")

        # 恢复时values会先返回已有状态；仅展示本次新产生的事件，完成但尚未归档的答案除外。
        seen = set()
        if resume:
            snapshot = await self.agent.aget_state(stream_kwargs["config"])
            if not snapshot.values:
                raise ValueError("找不到可恢复的检查点")
            if any(getattr(message, "name", None) == "fill_context_for_report" for message in snapshot.values.get("messages", [])):
                context["report"] = True
            if not snapshot.next:
                latest = snapshot.values.get("messages", [])[-1]
                if latest.type == "ai" and not latest.tool_calls and latest.content:
                    yield {"type": "answer", "content": latest.content.strip()}
                    return
                raise ValueError("检查点没有待继续步骤或有效最终回答")
            seen.update(message.id for message in snapshot.values.get("messages", []) if message.id)

        # 异步流式（MCP 工具仅支持异步调用）
        async for chunk in self.agent.astream(input_dict, **stream_kwargs):
            latest_message = chunk["messages"][-1]
            if latest_message.id and latest_message.id in seen:
                continue
            if latest_message.id:
                seen.add(latest_message.id)

            # report是请求级运行上下文；恢复时根据已保存工具轨迹重建，身份始终来自JWT。
            if any(getattr(message, "name", None) == "fill_context_for_report" for message in chunk["messages"]):
                context["report"] = True

            # 1. 模型发起工具调用 → 作为「动作」输出（把 ReAct 的思考动作化）
            if latest_message.type == "ai" and latest_message.tool_calls:
                for tc in latest_message.tool_calls:
                    name = tc.get("name", "unknown")
                    args = tc.get("args", {})
                    yield {"type": "action", "content": f"调用工具 {name}，参数 {args}"}

            # 2. 最终回答（AI 消息、无工具调用、content 非空）
            elif latest_message.type == "ai" and latest_message.content:
                yield {"type": "answer", "content": latest_message.content.strip()}

    async def checkpoints(self, execution_id, limit=30):
        """服务层授权后调用，返回有限检查点摘要；不暴露序列化原始状态或允许客户端指定线程。"""
        if not self.checkpointer:
            return []
        config = {"configurable": {"thread_id": execution_id}}
        return [{"checkpoint_id": row.config["configurable"]["checkpoint_id"],
                 "next": list(row.next), "created_at": row.created_at,
                 "message_count": len(row.values.get("messages", []))}
                async for row in self.agent.aget_state_history(config, limit=limit)]


if __name__ == '__main__':
    async def _main():
        """直接运行本文件时的调试入口，以显式用户1001执行一条报告请求并打印事件。"""
        agent = ReactAgent()

        async for item in agent.execute_stream([{"role": "user", "content": "给我生成我的使用报告"}], user_id="1001"):
            print(f"[{item['type']}] {item['content']}", flush=True)

    asyncio.run(_main())
