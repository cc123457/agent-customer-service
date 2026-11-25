import re
from datetime import datetime
from threading import Lock

from langchain_core.tools import tool
from langgraph.config import CONF, CONFIG_KEY_RUNTIME, get_config

from db.repository import RecordRepository
from utils.logger_handler import logger

_rag_service = None
_rag_lock = Lock()


def get_rag_service():
    """供 rag_summarize 获取进程内共享的 RAG 服务，首次使用时才初始化。

    用锁保护初始化，避免并发第一次调用重复构造 Chroma 客户端；
    返回服务对象，本方法本身不执行检索。
    """
    global _rag_service
    from rag.rag_service import RagSummarizeService
    with _rag_lock:
        if _rag_service is None:
            _rag_service = RagSummarizeService()
        return _rag_service

repo = RecordRepository()


def current_user_id() -> str:
    """从 Agent Runtime Context 读取并返回后端认证的用户 ID 字符串。

    get_user_id 和 authorize_user 共用；上下文缺失或身份无效抛 PermissionError。
    不信任用户文字，不随机选择用户，也不在这里查询数据库。
    """
    try:
        runtime = get_config()[CONF][CONFIG_KEY_RUNTIME]
        user_id = runtime.context.get("user_id")
    except (KeyError, AttributeError, RuntimeError) as error:
        raise PermissionError("缺少已认证的用户身份") from error
    if not isinstance(user_id, str) or not user_id.strip():
        raise PermissionError("缺少已认证的用户身份")
    return user_id


def authorize_user(user_id: str) -> str:
    """比较模型传入的 user_id 与运行身份，匹配则返回认证身份，否则拒绝。

    get_user_location 和 fetch_external_data 在访问数据库前调用；
    不一致抛 PermissionError，参数是字符串不代表它已经获得访问权限。
    """
    authenticated = current_user_id()
    if user_id != authenticated:
        raise PermissionError("仅允许查询当前登录用户的数据")
    return authenticated


@tool(description="从向量存储中检索参考资料")
def rag_summarize(query: str) -> str:
    """由 Agent 工具节点调用，输入查询词，返回混合检索后由模型总结的文字。

    通过 get_rag_service 按需获取服务；输出不是原始 Document 列表，
    工具内还会调用摘要模型，检索或模型异常由外层工具中间件处理。
    """
    return get_rag_service().rag_summarize(query)


@tool(description="获取指定用户所在城市的名称，以纯字符串形式返回；未查到返回空字符串")
def get_user_location(user_id: str) -> str:
    """输入模型提出的用户 ID，先授权再查询 SQLite，返回城市或空字符串。

    跨用户访问在 authorize_user 中拒绝，不能依赖提示词代替这里的检查。
    """
    return repo.get_user_city(authorize_user(user_id)) or ""


@tool(description="获取当前登录用户的ID，以纯字符串形式返回")
def get_user_id() -> str:
    """无工具参数，返回 Runtime Context 中的认证用户 ID。

    供 Agent 查询个人记录前使用；身份缺失即拒绝，不使用随机用户兜底。
    """
    return current_user_id()


@tool(description="获取系统当前月份，格式 YYYY-MM，以纯字符串形式返回")
def get_current_month() -> str:
    """无工具参数，返回服务器当前月份的 YYYY-MM 字符串，不查询模型。

    供未指定月份的报告查询使用；日期来自 datetime.now() 的服务器时区。
    """
    return datetime.now().strftime("%Y-%m")


@tool(description="从数据库中获取指定用户在指定月份的使用记录，以纯字符串形式返回，未检索到返回空字符串")
def fetch_external_data(user_id: str, month: str) -> str:
    """输入模型生成的用户 ID 与月份，先校验授权和 YYYY-MM，再查询个人记录。

    身份不符抛 PermissionError，非法月份抛 ValueError，均在数据库访问前拒绝。
    查到记录返回 Python 字典的字符串表示，未查到返回空字符串，不生成报告正文。
    """
    authenticated = authorize_user(user_id)
    if not re.fullmatch(r"[0-9]{4}-(0[1-9]|1[0-2])", month) or month.startswith("0000"):
        raise ValueError("月份必须是有效的 YYYY-MM")
    record = repo.get_record(authenticated, month)
    if record is None:
        logger.warning(f"[fetch_external_data]未能检索到用户：{user_id}在{month}的使用记录数据")
        return ""
    return str(record)


@tool(description="无入参，无返回值，调用后触发中间件自动为报告生成的场景动态注入上下文信息，为后续提示词切换提供上下文信息")
def fill_context_for_report() -> str:
    """无参数，返回报告切换确认文字，本函数不直接修改上下文。

    middleware.py 的 monitor_tool 在本工具成功后把 context['report'] 设为 True，
    后续模型调用由 report_prompt_switch 选择报告提示词。
    """
    return "fill_context_for_report已调用"
