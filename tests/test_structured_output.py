import asyncio
from unittest.mock import AsyncMock, Mock

import pytest
from langchain_core.messages import AIMessage

from services.structured_output import CustomerRequestExtractor, StructuredOutputError


def make_extractor(contents):
    model = Mock()
    bound = model.bind.return_value
    bound.ainvoke = AsyncMock(side_effect=[AIMessage(content=value) for value in contents])
    return CustomerRequestExtractor(model), model, bound


def test_valid_result_and_json_mode():
    extractor, model, bound = make_extractor([
        '{"intent":"troubleshooting","product_model":null,"summary":"无法充电"}'
    ])
    result = asyncio.run(extractor.extract("扫地机器人无法充电"))
    assert result.intent == "troubleshooting"
    assert result.product_model is None
    model.bind.assert_called_once_with(response_format={"type": "json_object"})
    assert bound.ainvoke.await_count == 1


@pytest.mark.parametrize("invalid", [
    "不是 JSON",
    '{"intent":"unknown","product_model":null,"summary":"查询"}',
    '{"intent":"other","product_model":123,"summary":"查询"}',
    '{"intent":"other","product_model":null,"summary":"查询","extra":true}',
    '{"intent":"other","summary":"查询"}',
])
def test_invalid_output_is_retried(invalid):
    extractor, _, bound = make_extractor([
        invalid, '{"intent":"other","product_model":null,"summary":"查询"}'
    ])
    result = asyncio.run(extractor.extract("查询"))
    assert result.summary == "查询"
    assert bound.ainvoke.await_count == 2


def test_retry_is_bounded():
    extractor, _, bound = make_extractor(["{}", "{}"])
    with pytest.raises(StructuredOutputError):
        asyncio.run(extractor.extract("查询"))
    assert bound.ainvoke.await_count == 2


def test_empty_input_never_calls_model():
    extractor, _, bound = make_extractor([])
    with pytest.raises(ValueError):
        asyncio.run(extractor.extract("  "))
    bound.ainvoke.assert_not_awaited()
