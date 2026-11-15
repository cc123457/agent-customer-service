import os
from abc import ABC, abstractmethod
from typing import Optional
from langchain_core.embeddings import Embeddings
from langchain_community.chat_models.tongyi import BaseChatModel
from langchain_community.embeddings import DashScopeEmbeddings
from langchain_community.chat_models.tongyi import ChatTongyi
from utils.config_handler import rag_conf
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI, OpenAIEmbeddings

load_dotenv(override=True)

class BaseModelFactory(ABC):
    @abstractmethod
    def generator(self) -> Optional[Embeddings | BaseChatModel]:
        pass


class ChatModelFactory(BaseModelFactory):
    def generator(self) -> Optional[Embeddings | BaseChatModel]:
        # 🚨 换成 ChatOpenAI
        return ChatOpenAI(
            model=rag_conf["chat_model_name"],
            api_key=os.getenv("DASHSCOPE_API_KEY"),
            base_url=os.getenv("DASHSCOPE_BASE_URL")
        )

class EmbeddingsFactory(BaseModelFactory):
    def generator(self) -> Optional[Embeddings | BaseChatModel]:
        return  OpenAIEmbeddings(
            model="text-embedding-v4",
            # 告诉它去读你命名的变量
            api_key=os.getenv("DASHSCOPE_API_KEY"),
            # 如果你 .env 里配了 BASE_URL，也可以这样读
            base_url=os.getenv("DASHSCOPE_BASE_URL"),
            # 🚨 加上这一行！关闭底层的 token 自动计算，直接发送原始字符串
            check_embedding_ctx_length=False
        )


chat_model = ChatModelFactory().generator()
embed_model = EmbeddingsFactory().generator()
