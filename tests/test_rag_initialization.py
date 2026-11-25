from concurrent.futures import ThreadPoolExecutor
from unittest.mock import Mock

from agent.tools import agent_tools


def test_concurrent_first_use_creates_one_rag_service(monkeypatch):
    import rag.rag_service as rag_module

    factory = Mock()
    monkeypatch.setattr(rag_module, "RagSummarizeService", factory)
    monkeypatch.setattr(agent_tools, "_rag_service", None)
    with ThreadPoolExecutor(max_workers=4) as pool:
        results = list(pool.map(lambda _: agent_tools.get_rag_service(), range(8)))
    factory.assert_called_once()
    assert all(result is results[0] for result in results)
