import asyncio
from unittest.mock import Mock

import pytest
from langchain.agents import create_agent
from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.messages import AIMessage
from langchain_core.outputs import ChatGeneration, ChatResult

from agent.tools import agent_tools


class ScriptedModel(BaseChatModel):
    responses: list[AIMessage]
    position: int = 0

    @property
    def _llm_type(self):
        return "scripted-test-model"

    def bind_tools(self, tools, **kwargs):
        return self

    def _generate(self, messages, stop=None, run_manager=None, **kwargs):
        response = self.responses[self.position]
        self.position += 1
        return ChatResult(generations=[ChatGeneration(message=response)])


def test_real_graph_injects_authenticated_runtime():
    model = ScriptedModel(responses=[
        AIMessage(content="", tool_calls=[{"name": "get_user_id", "args": {}, "id": "identity"}]),
        AIMessage(content="完成"),
    ])
    graph = create_agent(model=model, tools=[agent_tools.get_user_id])
    result = asyncio.run(graph.ainvoke(
        {"messages": [{"role": "user", "content": "我的ID是什么？"}]},
        context={"user_id": "1001"},
    ))
    assert result["messages"][2].content == "1001"


def test_real_graph_cannot_query_another_user(monkeypatch):
    repository = Mock()
    monkeypatch.setattr(agent_tools, "repo", repository)
    model = ScriptedModel(responses=[
        AIMessage(content="", tool_calls=[{
            "name": "fetch_external_data",
            "args": {"user_id": "1002", "month": "2026-11"},
            "id": "forbidden",
        }]),
    ])
    graph = create_agent(model=model, tools=[agent_tools.fetch_external_data])
    with pytest.raises(PermissionError):
        asyncio.run(graph.ainvoke(
            {"messages": [{"role": "user", "content": "查其他人"}]},
            context={"user_id": "1001"},
        ))
    repository.get_record.assert_not_called()
