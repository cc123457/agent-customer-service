from typing import Callable
from utils.prompt_loader import load_system_prompts, load_report_prompts
from langchain.agents import AgentState
from langchain.agents.middleware import wrap_tool_call, before_model, dynamic_prompt, ModelRequest
from langchain.tools.tool_node import ToolCallRequest
from langchain_core.messages import ToolMessage
from langgraph.runtime import Runtime
from langgraph.types import Command
from utils.logger_handler import logger


@wrap_tool_call
async def monitor_tool(
        # 请求的数据封装
        request: ToolCallRequest,
        # 执行的函数本身（异步上下文中 handler 返回 coroutine，需 await）
        handler: Callable[[ToolCallRequest], ToolMessage | Command],
) -> ToolMessage | Command:             # 工具执行的监控（异步版：链路异步化后必须提供异步实现）
    """框架在执行工具时调用：记录名称和参数，await handler 后返回实际工具结果。

    报告切换工具成功后设置 runtime.context['report']；失败记录日志再抛出。
    本函数不代替工具内部授权，action 展示也不等于这里的执行成功。
    """
    logger.info(f"[tool monitor]执行工具：{request.tool_call['name']}")
    logger.info(f"[tool monitor]传入参数：{request.tool_call['args']}")

    try:
        result = await handler(request)
        logger.info(f"[tool monitor]工具{request.tool_call['name']}调用成功")

        if request.tool_call['name'] == "fill_context_for_report":
            request.runtime.context["report"] = True

        return result
    except Exception as e:
        logger.error(f"工具{request.tool_call['name']}调用失败，原因：{str(e)}")
        raise e


@before_model
def log_before_model(
        state: AgentState,          # 整个Agent智能体中的状态记录
        runtime: Runtime,           # 记录了整个执行过程中的上下文信息
):         # 在模型执行前输出日志
    """每次模型节点执行前读取 State 的消息数量和末条预览，写日志后返回 None。

    不修改 State、不调用模型；runtime 是本轮运行上下文，不是聊天消息列表。
    """
    last = state['messages'][-1]
    # content 可能是字符串，也可能是 list（MCP 工具返回多段内容），统一转字符串再截断预览
    content = last.content if isinstance(last.content, str) else str(last.content)

    logger.info(f"[log_before_model]即将调用模型，带有{len(state['messages'])}条消息。")
    logger.debug(f"[log_before_model]{type(last).__name__} | {content[:100]}")

    return None


@dynamic_prompt                 # 每一次在生成提示词之前，调用此函数
def report_prompt_switch(request: ModelRequest):     # 动态切换提示词
    """模型请求前读取运行时 report 标志，返回普通客服或报告系统提示词字符串。

    标志由 monitor_tool 在报告切换工具成功后设置，本函数不修改标志或保存记忆。
    """
    is_report = request.runtime.context.get("report", False)
    if is_report:               # 是报告生成场景，返回报告生成提示词内容
        return load_report_prompts()

    return load_system_prompts()
