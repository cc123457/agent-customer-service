"""可观测性：Langfuse 链路追踪（可选启用）。

未配置 LANGFUSE key 时返回 None，程序照常运行（优雅降级，不影响功能）。
"""
import os

from dotenv import load_dotenv

from utils.path_tool import get_abs_path

load_dotenv(get_abs_path(".env"), override=False)


def get_langfuse_handler():
    """若配置了 Langfuse 凭据则返回 CallbackHandler，否则返回 None（不启用追踪）"""
    public_key = os.getenv("LANGFUSE_PUBLIC_KEY")
    secret_key = os.getenv("LANGFUSE_SECRET_KEY")
    if not public_key or not secret_key:
        return None

    from langfuse.langchain import CallbackHandler
    return CallbackHandler()
