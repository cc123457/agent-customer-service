import random
from datetime import datetime

from langchain_core.tools import tool
from langgraph.config import CONF, CONFIG_KEY_RUNTIME, get_config

from db.repository import RecordRepository
from rag.rag_service import RagSummarizeService
from utils.logger_handler import logger

rag = RagSummarizeService()

repo = RecordRepository()


@tool(description="从向量存储中检索参考资料")
def rag_summarize(query: str) -> str:
    return rag.rag_summarize(query)


@tool(description="获取指定用户所在城市的名称，以纯字符串形式返回；未查到返回空字符串")
def get_user_location(user_id: str) -> str:
    return repo.get_user_city(user_id) or ""


@tool(description="获取当前登录用户的ID，以纯字符串形式返回")
def get_user_id() -> str:
    # 从 runtime context 取当前登录用户（由 JWT 鉴权注入）
    try:
        runtime = get_config()[CONF][CONFIG_KEY_RUNTIME]
        uid = runtime.context.get("user_id")
        if uid:
            return uid
    except Exception:
        pass
    # 兜底：拿不到登录态时（如本地调试）随机取一个
    return random.choice(repo.get_user_ids())


@tool(description="获取系统当前月份，格式 YYYY-MM，以纯字符串形式返回")
def get_current_month() -> str:
    return datetime.now().strftime("%Y-%m")


@tool(description="从数据库中获取指定用户在指定月份的使用记录，以纯字符串形式返回，未检索到返回空字符串")
def fetch_external_data(user_id: str, month: str) -> str:
    record = repo.get_record(user_id, month)
    if record is None:
        logger.warning(f"[fetch_external_data]未能检索到用户：{user_id}在{month}的使用记录数据")
        return ""
    return str(record)


@tool(description="无入参，无返回值，调用后触发中间件自动为报告生成的场景动态注入上下文信息，为后续提示词切换提供上下文信息")
def fill_context_for_report():
    return "fill_context_for_report已调用"
