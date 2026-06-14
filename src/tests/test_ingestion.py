"""Unit tests for parent-child chunking and parent-context retrieval."""
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from src.core.config import settings
from src.services.ingestion import DocumentIngestionService, split_text


def test_parent_chunks_within_size_limit():
    """Every parent chunk must be at most PARENT_CHUNK_SIZE characters."""
    text = "x" * 10_000
    parents = split_text(text, settings.PARENT_CHUNK_SIZE, 200)
    assert parents, "expected at least one parent chunk"
    assert all(len(p) <= 1500 for p in parents)


def test_child_chunks_within_size_limit():
    """Every child chunk must be at most CHILD_CHUNK_SIZE characters."""
    text = "y" * 1500
    children = split_text(text, settings.CHILD_CHUNK_SIZE, 50)
    assert all(len(c) <= 400 for c in children)


@pytest.mark.asyncio
async def test_child_chunks_carry_parent_id_metadata():
    """Upserted child points must include parent_id and parent_text in payload."""
    service = DocumentIngestionService()
    captured_points = []

    fake_client = MagicMock()
    fake_client.collection_exists.return_value = True
    fake_client.upsert.side_effect = lambda collection_name, points: captured_points.extend(points)

    fake_embeddings = MagicMock()
    fake_embeddings.embed_documents.side_effect = lambda texts: [
        [0.0] * 384 for _ in texts
    ]
    service._embeddings = fake_embeddings

    with patch("src.services.ingestion.get_qdrant_client", return_value=fake_client):
        result = await service.ingest(b"hello world " * 500, "test.txt")

    assert result["status"] == "indexed"
    assert result["parent_count"] > 0
    assert result["child_count"] >= result["parent_count"]
    assert captured_points, "expected points to be upserted"
    for point in captured_points:
        assert "parent_id" in point.payload
        assert "parent_text" in point.payload
        assert point.payload["filename"] == "test.txt"
        assert "chunk_index" in point.payload


@pytest.mark.asyncio
async def test_retrieve_returns_parent_text_not_child_text():
    """retrieve() must surface the full parent context, deduplicated by parent_id."""
    service = DocumentIngestionService()

    def make_hit(parent_id: str, parent_text: str, child_text: str) -> MagicMock:
        hit = MagicMock()
        hit.payload = {
            "parent_id": parent_id,
            "parent_text": parent_text,
            "child_text": child_text,
        }
        return hit

    fake_client = MagicMock()
    fake_client.query_points.return_value.points = [
        make_hit("p1", "full parent one", "child one"),
        make_hit("p1", "full parent one", "child one b"),  # duplicate parent
        make_hit("p2", "full parent two", "child two"),
    ]
    fake_embeddings = MagicMock()
    fake_embeddings.aembed_query = AsyncMock(return_value=[0.0] * 1536)
    service._embeddings = fake_embeddings

    with patch("src.services.ingestion.get_qdrant_client", return_value=fake_client):
        contexts = await service.retrieve("query", top_k=5)

    assert contexts == ["full parent one", "full parent two"]
    assert "child one" not in contexts
