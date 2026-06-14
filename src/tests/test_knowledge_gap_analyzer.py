"""Tests for Feature 3: Knowledge Gap Analyzer."""
import json
from unittest.mock import MagicMock

from src.services.knowledge_gap_analyzer import _cache_key, analyze_knowledge_gaps


def test_cache_key_consistent():
    key1 = _cache_key("what are my skills", ["chunk a", "chunk b"])
    key2 = _cache_key("what are my skills", ["chunk a", "chunk b"])
    assert key1 == key2


def test_cache_key_order_independent():
    key1 = _cache_key("test", ["chunk a", "chunk b"])
    key2 = _cache_key("test", ["chunk b", "chunk a"])
    assert key1 == key2


def test_cache_key_different_questions():
    key1 = _cache_key("what are my skills")
    key2 = _cache_key("what is my GPA")
    assert key1 != key2


def test_cache_key_different_chunks():
    key1 = _cache_key("test", ["resume content"])
    key2 = _cache_key("test", ["transcript content"])
    assert key1 != key2


def test_malformed_json_returns_none():
    mock_llm = MagicMock()
    mock_response = MagicMock()
    mock_response.content = "not json at all"
    mock_llm.invoke.return_value = mock_response

    result = analyze_knowledge_gaps("test", ["chunk"], "answer", mock_llm)
    assert result is None


def test_valid_response_parsed_correctly():
    valid_json = json.dumps({
        "missing_info": ["GPA"],
        "suggested_documents": ["Transcript"],
        "completeness_score": 5,
        "gap_summary": "Missing academic records.",
    })
    mock_llm = MagicMock()
    mock_response = MagicMock()
    mock_response.content = valid_json
    mock_llm.invoke.return_value = mock_response

    result = analyze_knowledge_gaps("test", ["resume chunk"], "some answer", mock_llm)
    assert result is not None
    assert result["completeness_score"] == 5
    assert result["missing_info"] == ["GPA"]
    assert result["from_cache"] is False


def test_score_clamped_to_valid_range():
    valid_json = json.dumps({
        "missing_info": [],
        "suggested_documents": [],
        "completeness_score": 15,  # out of range
        "gap_summary": "Complete answer.",
    })
    mock_llm = MagicMock()
    mock_response = MagicMock()
    mock_response.content = valid_json
    mock_llm.invoke.return_value = mock_response

    result = analyze_knowledge_gaps("unique_clamp_test_question_xyz", ["chunk"], "answer", mock_llm)
    assert result is not None
    assert result["completeness_score"] == 10
