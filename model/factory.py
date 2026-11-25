import os
from abc import ABC, abstractmethod
from typing import Optional
from langchain_core.embeddings import Embeddings
from langchain_community.chat_models.tongyi import BaseChatModel
from utils.config_handler import rag_conf
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from model.usage import AsyncMeteredEmbeddingClient, MeteredEmbeddingClient

load_dotenv(override=False)

class BaseModelFactory(ABC):
    @abstractmethod
    def generator(self) -> Optional[Embeddings | BaseChatModel]:
        """模型工厂的抽象创建入口，由子类返回聊天或向量模型对象，不执行推理。"""
        pass


class ChatModelFactory(BaseModelFactory):
    def generator(self) -> Optional[Embeddings | BaseChatModel]:
        """读取 rag.yml 模型名和环境中的凭据/地址，返回 OpenAI 兼容聊天客户端。

        模块底部创建 chat_model，供 Agent、摘要和 JSON 提取器复用；调用对象后才发起推理。
        """
        # 🚨 换成 ChatOpenAI
        return ChatOpenAI(
            model=rag_conf["chat_model_name"],
            api_key=os.getenv("DASHSCOPE_API_KEY"),
            base_url=os.getenv("DASHSCOPE_BASE_URL")
        )

class EmbeddingsFactory(BaseModelFactory):
    def generator(self) -> Optional[Embeddings | BaseChatModel]:
        """按配置或显式环境覆盖创建客户端，供 Chroma 入库和查询向量化使用。

        凭据与接口地址来自环境；关闭本地 Token 自动切分以发送原始字符串，不执行向量化。
        """
        model = OpenAIEmbeddings(
            model=os.getenv("EMBEDDING_MODEL_NAME") or rag_conf["embedding_model_name"],
            # 告诉它去读你命名的变量
            api_key=os.getenv("DASHSCOPE_API_KEY"),
            # 如果你 .env 里配了 BASE_URL，也可以这样读
            base_url=os.getenv("DASHSCOPE_BASE_URL"),
            # 🚨 加上这一行！关闭底层的 token 自动计算，直接发送原始字符串
            check_embedding_ctx_length=False,
            # 当前兼容接口每批最多10条；索引可提交更大的列表，由SDK逐批发送。
            chunk_size=10,
            dimensions=rag_conf.get("embedding_dimensions"),
        )
        model.client = MeteredEmbeddingClient(model.client, model.model)
        model.async_client = AsyncMeteredEmbeddingClient(model.async_client, model.model)
        return model


chat_model = ChatModelFactory().generator()
embed_model = EmbeddingsFactory().generator()
