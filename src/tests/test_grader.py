"""Unit tests for grade_documents node and the loop guardrail edge."""
from unittest.mock import MagicMock

import pytest

import src.agents.nodes as nodes
from src.agents.edges import decide_after_grading


def _mock_score_response(score: str) -> MagicMock:
    """Build a fake AIMessage-like grading response."""
    response = MagicMock()
    response.content = score
    response.usage_metadata = {"input_tokens": 100, "output_tokens": 3}
    return response


@pytest.fixture
def mock_llm(monkeypatch):
    """Replace the module-level GPT-4o client with a controllable mock."""
    llm = MagicMock()
    monkeypatch.setattr(nodes, "_llm", llm)
    return llm


def test_low_score_doc_is_filtered(mock_llm):
    """A document graded 0.3 falls below the 0.6 threshold and is removed."""
    mock_llm.invoke.side_effect = [_mock_score_response("0.9"), _mock_score_response("0.3")]
    state = {"question": "q", "documents": ["relevant doc", "irrelevant doc"]}
    result = nodes.grade_documents(state)
    assert result["documents"] == ["relevant doc"]
    assert result["relevance_scores"] == [0.9, 0.3]
    assert result["web_search"] is False


def test_all_docs_filtered_sets_web_search(mock_llm):
    """If every document is filtered, web_search must be True."""
    mock_llm.invoke.side_effect = [_mock_score_response("0.1"), _mock_score_response("0.2")]
    state = {"question": "q", "documents": ["doc a", "doc b"]}
    result = nodes.grade_documents(state)
    assert result["documents"] == []
    assert result["web_search"] is True


def test_unparseable_score_treated_as_zero(mock_llm):
    """A non-numeric grade is treated as 0.0 and filtered."""
    mock_llm.invoke.side_effect = [_mock_score_response("not a number")]
    result = nodes.grade_documents({"question": "q", "documents": ["doc"]})
    assert result["relevance_scores"] == [0.0]
    assert result["web_search"] is True


def test_guardrail_forces_web_search_at_max_loops():
    """loop_count >= 2 with no relevant docs must exit the rewrite loop to web search."""
    state = {"web_search": True, "loop_count": 2}
    assert decide_after_grading(state) == "web_search"


def test_below_max_loops_rewrites_query():
    """With no relevant docs and loops remaining, the query gets rewritten."""
    state = {"web_search": True, "loop_count": 1}
    assert decide_after_grading(state) == "transform_query"


def test_relevant_docs_go_to_generate():
    """If documents passed grading, the graph proceeds to generation."""
    state = {"web_search": False, "loop_count": 0}
    assert decide_after_grading(state) == "generate"
