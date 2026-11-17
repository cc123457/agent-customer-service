from unittest.mock import Mock, mock_open

import pytest
from langchain_core.documents import Document

from rag.vector_store import HybridRetriever, VectorStoreService


def make_service(texts):
    service = VectorStoreService.__new__(VectorStoreService)
    service.vector_store = Mock()
    service.vector_store.get.return_value = {
        "documents": texts, "metadatas": [{} for text in texts],
    }
    service.vector_store.similarity_search.return_value = []
    service._bm25_retriever = None
    return service


def test_empty_and_blank_knowledge_base():
    for texts in ([], ["", "  "]):
        service = make_service(texts)
        assert HybridRetriever(service).invoke("filter") == []
        assert service._bm25_retriever is None


def test_blank_query_and_invalid_limit():
    service = make_service([])
    assert service.hybrid_search("  ") == []
    service.vector_store.similarity_search.assert_not_called()
    with pytest.raises(ValueError):
        service.hybrid_search("filter", top_n=0)


def test_rrf_rewards_agreement_and_removes_duplicates():
    shared = Document(page_content="shared")
    vector_only = Document(page_content="vector")
    keyword_only = Document(page_content="keyword")
    result = VectorStoreService._rrf_fusion(
        [[vector_only, shared], [keyword_only, shared]], top_n=3
    )
    assert result[0].page_content == "shared"
    assert len({doc.page_content for doc in result}) == 3


def test_chinese_keyword_recall_and_custom_limit():
    service = make_service([
        "滤网 清洗 滤网 维护", "拖布 拖地 清洁", "电池 充电 续航",
        "地图 建图 导航", "滚刷 毛发 缠绕",
    ])
    result = service.hybrid_search("滤网", top_n=4)
    assert result[0].page_content == "滤网 清洗 滤网 维护"
    assert len(result) == 4
    cached = service._bm25_retriever
    service.hybrid_search("拖布")
    assert service._bm25_retriever is cached


def test_successful_ingestion_invalidates_bm25(monkeypatch):
    import rag.vector_store as vector_module

    service = make_service(["old document"])
    previous = service._get_bm25_retriever()
    service.spliter = Mock()
    service.spliter.split_documents.return_value = [Document(page_content="new document")]
    monkeypatch.setattr(vector_module, "listdir_with_allowed_type", lambda *args: ["new.txt"])
    monkeypatch.setattr(vector_module, "get_file_md5_hex", lambda path: "new-md5")
    monkeypatch.setattr(vector_module, "txt_loader", lambda path: [Document(page_content="new document")])
    monkeypatch.setattr("builtins.open", mock_open(read_data=""))
    service.load_document()
    service.vector_store.add_documents.assert_called_once()
    assert service._bm25_retriever is None
    service.vector_store.get.return_value = {"documents": ["new document"], "metadatas": [{}]}
    assert service._get_bm25_retriever() is not previous
    assert service._bm25_retriever.docs[0].page_content == "new document"


def test_keyword_branch_recovers_vector_miss():
    service = make_service(["滤网 清洗 维护", "拖布 拖地 清洁", "电池 充电 续航"])
    service.vector_store.similarity_search.return_value = [Document(page_content="拖布 拖地 清洁")]
    vector_result = service.vector_store.similarity_search("滤网")
    hybrid_result = service.hybrid_search("滤网", top_n=2)
    assert not any("滤网" in doc.page_content for doc in vector_result)
    assert any("滤网" in doc.page_content for doc in hybrid_result)
