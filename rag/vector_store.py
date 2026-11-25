from langchain_chroma import Chroma
from langchain_core.documents import Document
from utils.config_handler import chroma_conf
from model.factory import embed_model
from langchain_text_splitters import RecursiveCharacterTextSplitter
from utils.path_tool import get_abs_path
from utils.file_handler import pdf_loader, txt_loader, listdir_with_allowed_type, get_file_md5_hex
from utils.logger_handler import logger
from langchain_community.retrievers import BM25Retriever
import jieba
import os
from model.embedding_space import embedding_space


class VectorStoreService:
    def __init__(self):
        """每个向量空间使用独立集合及入库清单，模型更换后重新入库，不删除旧库。"""
        self.space = embedding_space(embed_model)
        self.md5_store = get_abs_path(chroma_conf["md5_hex_store"]) + "." + self.space
        self.vector_store = Chroma(
            collection_name=chroma_conf["collection_name"] + "_" + self.space,
            embedding_function=embed_model,
            persist_directory=get_abs_path(chroma_conf["persist_directory"]),
        )

        self.spliter = RecursiveCharacterTextSplitter(
            chunk_size=chroma_conf["chunk_size"],
            chunk_overlap=chroma_conf["chunk_overlap"],
            separators=chroma_conf["separators"],
            length_function=len,
        )

        self._bm25_retriever = None   # BM25 关键词检索器（懒加载 + 缓存）

    def get_retriever(self):
        """返回混合检索器（向量 + BM25 关键词，RRF 融合）"""
        return HybridRetriever(self)

    def _load_all_documents(self) -> list[Document]:
        """从向量库取回所有分片文档，用于构建 BM25 关键词索引"""
        data = self.vector_store.get(include=["documents", "metadatas"])
        documents = data.get("documents") or []
        metadatas = data.get("metadatas") or [{}] * len(documents)
        docs = []
        for text, meta in zip(documents, metadatas):
            docs.append(Document(page_content=text, metadata=meta or {}))
        return docs

    def _get_bm25_retriever(self) -> BM25Retriever | None:
        """BM25 关键词检索器（中文用 jieba 分词，懒加载 + 缓存）"""
        if self._bm25_retriever is None:
            documents = [doc for doc in self._load_all_documents() if doc.page_content.strip()]
            if not documents:
                return None
            self._bm25_retriever = BM25Retriever.from_documents(
                documents,
                preprocess_func=jieba.lcut,   # 中文分词：默认按空格会整句当一词，BM25 失效
            )
            self._bm25_retriever.k = chroma_conf["k"]
        return self._bm25_retriever

    def hybrid_search(self, query: str, top_n: int = None) -> list[Document]:
        """混合检索：向量相似度 + BM25 关键词，RRF 融合"""
        top_n = chroma_conf["k"] if top_n is None else top_n
        if top_n < 1:
            raise ValueError("top_n must be positive")
        if not query.strip():
            return []
        vector_docs = self.vector_store.similarity_search(query, k=top_n)
        bm25_retriever = self._get_bm25_retriever()
        bm25_docs = []
        if bm25_retriever is not None:
            bm25_retriever.k = top_n
            bm25_docs = bm25_retriever.invoke(query)
        return self._rrf_fusion([vector_docs, bm25_docs], top_n=top_n)

    @staticmethod
    def _rrf_fusion(doc_lists: list, k: int = 60, top_n: int = 3) -> list[Document]:
        """RRF 融合：score = Σ 1/(k + rank)，按文档内容去重累加排名倒数"""
        fused = {}
        for docs in doc_lists:
            for rank, doc in enumerate(docs):
                key = doc.page_content
                if key not in fused:
                    fused[key] = [doc, 0.0]
                fused[key][1] += 1.0 / (k + rank + 1)
        ranked = sorted(fused.values(), key=lambda x: x[1], reverse=True)
        return [doc for doc, _ in ranked[:top_n]]

    def load_document(self):
        """
        从数据文件夹内读取数据文件，转为向量存入向量库
        要计算文件的MD5做去重
        :return: None
        """

        def check_md5_hex(md5_for_check: str):
            if not os.path.exists(self.md5_store):
                # 创建文件
                open(self.md5_store, "w", encoding="utf-8").close()
                return False            # md5 没处理过

            with open(self.md5_store, "r", encoding="utf-8") as f:
                for line in f.readlines():
                    line = line.strip()
                    if line == md5_for_check:
                        return True     # md5 处理过

                return False            # md5 没处理过

        def save_md5_hex(md5_for_check: str):
            with open(self.md5_store, "a", encoding="utf-8") as f:
                f.write(md5_for_check + "\n")

        def get_file_documents(read_path: str):
            """
            加载文件
            :param: read_path
            :return: list[Document]
            """
            if read_path.endswith("txt"):
                return txt_loader(read_path)

            if read_path.endswith("pdf"):
                return pdf_loader(read_path)

            return []

        allowed_files_path: list[str] = listdir_with_allowed_type(
            get_abs_path(chroma_conf["data_path"]),
            tuple(chroma_conf["allow_knowledge_file_type"]),
        )

        for path in allowed_files_path:
            # 获取文件的MD5
            md5_hex = get_file_md5_hex(path)

            if check_md5_hex(md5_hex):
                logger.info(f"[加载知识库]{path}内容已经存在知识库内，跳过")
                continue

            try:
                documents: list[Document] = get_file_documents(path)

                if not documents:
                    logger.warning(f"[加载知识库]{path}内没有有效文本内容，跳过")
                    continue

                split_document: list[Document] = self.spliter.split_documents(documents)

                if not split_document:
                    logger.warning(f"[加载知识库]{path}分片后没有有效文本内容，跳过")
                    continue

                # 将内容存入向量库
                self.vector_store.add_documents(split_document)
                self._bm25_retriever = None

                # 记录这个已经处理好的文件的md5，避免下次重复加载
                save_md5_hex(md5_hex)

                logger.info(f"[加载知识库]{path} 内容加载成功")
            except Exception as e:
                # exc_info为True会记录详细的报错堆栈，如果为False仅记录报错信息本身
                logger.error(f"[加载知识库]{path}加载失败：{str(e)}", exc_info=True)
                continue


class HybridRetriever:
    """混合检索器包装：暴露 invoke 接口，与 LangChain retriever 调用方式一致"""

    def __init__(self, service: VectorStoreService):
        self._service = service

    def invoke(self, query: str) -> list[Document]:
        return self._service.hybrid_search(query)


if __name__ == '__main__':
    vs = VectorStoreService()

    vs.load_document()

    retriever = vs.get_retriever()

    res = retriever.invoke("迷路")
    for r in res:
        print(r.page_content)
        print("-"*20)


