"""Tests for Feature 2: Answer Versioning."""
import pytest

from src.agents.nodes import _compute_answer_improvement


def test_compute_answer_improvement_detects_growth():
    versions = [
        {"draft_answer": "Short answer.", "retrieval_quality": 0.3},
        {
            "draft_answer": "Much longer and more detailed answer with specific information about the candidate's skills and experience.",
            "retrieval_quality": 0.75,
        },
    ]
    result = _compute_answer_improvement(versions)
    assert result["improved"] is True
    assert result["quality_delta"] > 0


def test_compute_answer_improvement_single_loop():
    versions = [{"draft_answer": "Answer.", "retrieval_quality": 0.8}]
    result = _compute_answer_improvement(versions)
    assert result["improved"] is False
    assert result["reason"] == "single_loop"


def test_compute_answer_improvement_no_change():
    versions = [
        {"draft_answer": "The candidate has Python skills.", "retrieval_quality": 0.6},
        {"draft_answer": "The candidate has Python skills.", "retrieval_quality": 0.6},
    ]
    result = _compute_answer_improvement(versions)
    assert result["improved"] is False


def test_compute_answer_improvement_quality_only():
    versions = [
        {"draft_answer": "The answer is here for testing purposes.", "retrieval_quality": 0.2},
        {"draft_answer": "The answer is here for testing purposes extended.", "retrieval_quality": 0.8},
    ]
    result = _compute_answer_improvement(versions)
    assert result["quality_delta"] == pytest.approx(0.6, abs=0.01)
