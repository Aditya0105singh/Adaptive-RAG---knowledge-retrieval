"""Tests for Feature 1: Grounding Checker."""
from unittest.mock import MagicMock

from src.services.grounding_checker import check_answer_grounding, split_into_sentences


def test_split_sentences_basic():
    text = "The candidate has Python skills. He studied at MIT. He may be good at math."
    sentences = split_into_sentences(text, min_len=5)
    assert len(sentences) == 3


def test_split_sentences_filters_short():
    text = "OK. The candidate has extensive Python and AWS experience. Great."
    sentences = split_into_sentences(text, min_len=20)
    assert len(sentences) == 1
    assert "Python" in sentences[0]


def test_max_sentences_guard():
    long_answer = ". ".join([f"Sentence number {i} about the topic in detail" for i in range(20)])
    mock_llm = MagicMock()

    result = check_answer_grounding(
        answer=long_answer,
        retrieved_chunks=["chunk"],
        llm=mock_llm,
        max_sentences=5,
    )
    assert result is not None
    assert result["skipped"] is True
    # LLM should never have been called
    mock_llm.invoke.assert_not_called()


def _make_llm_returning(label: str) -> MagicMock:
    mock_llm = MagicMock()
    mock_response = MagicMock()
    mock_response.content = label
    mock_llm.invoke.return_value = mock_response
    return mock_llm


def test_grounded_classification():
    mock_llm = _make_llm_returning("GROUNDED")
    result = check_answer_grounding(
        answer="The candidate has Python skills and AWS experience in cloud computing.",
        retrieved_chunks=["The candidate has Python skills and AWS experience."],
        llm=mock_llm,
        max_sentences=12,
    )
    assert result is not None
    assert not result["skipped"]
    assert result["summary"]["trust_level"] in {"HIGH", "MODERATE", "LOW"}
    assert all(r["label"] in {"GROUNDED", "INFERRED", "UNGROUNDED"} for r in result["results"])


def test_invalid_label_defaults_to_inferred():
    mock_llm = _make_llm_returning("I cannot determine this classification")
    result = check_answer_grounding(
        answer="The candidate has Python skills and experience with various cloud platforms.",
        retrieved_chunks=["chunk"],
        llm=mock_llm,
    )
    assert result is not None
    assert all(r["label"] == "INFERRED" for r in result["results"])


def test_trust_score_calculation():
    call_count = 0

    def side_effect(messages):
        nonlocal call_count
        mock_response = MagicMock()
        # Alternate GROUNDED / UNGROUNDED
        mock_response.content = "GROUNDED" if call_count % 2 == 0 else "UNGROUNDED"
        call_count += 1
        return mock_response

    mock_llm = MagicMock()
    mock_llm.invoke.side_effect = side_effect

    result = check_answer_grounding(
        answer="First sentence is grounded here. Second sentence is ungrounded here.",
        retrieved_chunks=["First sentence is grounded here."],
        llm=mock_llm,
        min_sentence_len=5,
    )
    assert result is not None
    summary = result["summary"]
    assert summary["trust_score"] == 0.5
    assert summary["trust_level"] == "MODERATE"
