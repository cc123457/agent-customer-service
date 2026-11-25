from types import SimpleNamespace
from unittest.mock import Mock

import pytest

from agent.tools import agent_tools


@pytest.fixture
def authorized(monkeypatch):
    repository = Mock()
    repository.get_user_city.return_value = "测试城市"
    repository.get_record.return_value = {"效率": "测试记录"}
    monkeypatch.setattr(agent_tools, "repo", repository)
    monkeypatch.setattr(agent_tools, "get_config", lambda: {
        agent_tools.CONF: {agent_tools.CONFIG_KEY_RUNTIME: SimpleNamespace(context={"user_id": "1001"})},
    })
    return repository


def test_identity_comes_from_runtime(authorized):
    assert agent_tools.get_user_id.invoke({}) == "1001"


@pytest.mark.parametrize("tool,args", [
    (agent_tools.get_user_location, {"user_id": "1002"}),
    (agent_tools.fetch_external_data, {"user_id": "1002", "month": "2026-11"}),
])
def test_cross_user_query_never_reaches_database(authorized, tool, args):
    with pytest.raises(PermissionError):
        tool.invoke(args)
    authorized.get_record.assert_not_called()
    authorized.get_user_city.assert_not_called()


def test_no_runtime_never_selects_random_user(monkeypatch):
    monkeypatch.setattr(agent_tools, "get_config", lambda: {})
    with pytest.raises(PermissionError):
        agent_tools.get_user_id.invoke({})


@pytest.mark.parametrize("month", ["2026-99", "2026-00", "2026-1", "0000-01", "随便"])
def test_invalid_month_rejected_before_database(authorized, month):
    with pytest.raises(ValueError):
        agent_tools.fetch_external_data.invoke({"user_id": "1001", "month": month})
    authorized.get_record.assert_not_called()


def test_valid_query(authorized):
    assert "测试记录" in agent_tools.fetch_external_data.invoke({"user_id": "1001", "month": "2026-11"})
    authorized.get_record.assert_called_once_with("1001", "2026-11")
