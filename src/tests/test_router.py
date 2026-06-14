"""Unit tests for the route_question node with mocked GPT-4o responses."""
from unittest.mock import MagicMock

import pytest

import src.agents.nodes as nodes


def _mock_llm_response(content: str) -> MagicMock:
    """Build a fake AIMessage-like object with content and usage metadata."""
    response = MagicMock()
    response.content = content
    response.usage_metadata = {"input_tokens": 50, "output_tokens": 2}
    return response


@pytest.fixture
def mock_llm(monkeypatch):
    """Replace the module-level GPT-4o client with a controllable mock."""
    llm = MagicMock()
    monkeypatch.setattr(nodes, "_llm", llm)
    return llm


def test_greeting_routes_to_general(mock_llm):
    """'Hi there' should be classified as general chat."""
    mock_llm.invoke.return_value = _mock_llm_response("general")
    result = nodes.route_question({"question": "Hi there", "token_usage": {}})
    assert result["route_taken"] == "general"


def test_hr_policy_routes_to_index(mock_llm):
    """A knowledge-base question should route to the vector index."""
    mock_llm.invoke.return_value = _mock_llm_response("index")
    result = nodes.route_question(
        {"question": "What does our HR policy say?", "token_usage": {}}
    )
    assert result["route_taken"] == "index"


def test_news_routes_to_search(mock_llm):
    """A current-events question should route to web search."""
    mock_llm.invoke.return_value = _mock_llm_response("search")
    result = nodes.route_question({"question": "Latest news about AI?", "token_usage": {}})
    assert result["route_taken"] == "search"


def test_invalid_route_falls_back_to_index(mock_llm):
    """An unexpected classifier output should fall back to 'index'."""
    mock_llm.invoke.return_value = _mock_llm_response("banana")
    result = nodes.route_question({"question": "Anything", "token_usage": {}})
    assert result["route_taken"] == "index"


def test_route_accumulates_token_usage(mock_llm):
    """Token usage from the routing call should be merged into state."""
    mock_llm.invoke.return_value = _mock_llm_response("general")
    result = nodes.route_question(
        {"question": "Hi", "token_usage": {"prompt": 10, "completion": 5}}
    )
    assert result["token_usage"] == {"prompt": 60, "completion": 7}
