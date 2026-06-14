"""Parent-child document chunking, embedding, and Qdrant indexing."""
import asyncio
import io
import uuid
import xml.etree.ElementTree as ET
import zipfile
from typing import List

from pypdf import PdfReader
from qdrant_client.models import Distance, PointStruct, VectorParams

from src.core.config import settings
from src.core.database import get_qdrant_client
from src.core.logging import get_logger
from src.services.embeddings import get_embeddings

logger = get_logger(__name__)

EMBED_BATCH_SIZE = 90  # Cohere embed API hard limit is 96 texts per request


def split_text(text: str, chunk_size: int, overlap: int) -> List[str]:
    """Split text into character chunks of chunk_size with the given overlap."""
    if chunk_size <= overlap:
        raise ValueError("chunk_size must be greater than overlap")
    chunks: List[str] = []
    start = 0
    while start < len(text):
        chunk = text[start : start + chunk_size].strip()
        if chunk:
            chunks.append(chunk)
        start += chunk_size - overlap
    return chunks


class DocumentIngestionService:
    """Ingests PDF/TXT files into Qdrant using parent-child chunking."""

    def _ensure_collection(self, collection_name: str) -> None:
        """Create the Qdrant collection if it does not already exist."""
        client = get_qdrant_client()
        try:
            if not client.collection_exists(collection_name):
                client.create_collection(
                    collection_name=collection_name,
                    vectors_config=VectorParams(
                        size=settings.EMBED_DIM, distance=Distance.COSINE
                    ),
                )
                logger.info("qdrant_collection_created", collection=collection_name)
        except Exception as exc:
            logger.error("qdrant_collection_ensure_failed", collection=collection_name, error=str(exc))
            raise

    @staticmethod
    def _parse_text(file_bytes: bytes, filename: str) -> str:
        """Extract raw text from PDF, TXT, MD, CSV, or DOCX bytes."""
        name = filename.lower()
        if name.endswith(".pdf"):
            reader = PdfReader(io.BytesIO(file_bytes))
            return "\n".join(page.extract_text() or "" for page in reader.pages)
        if name.endswith(".docx"):
            # .docx is a ZIP archive; text lives in word/document.xml w:t nodes.
            ns = "http://schemas.openxmlformats.org/wordprocessingml/2006/main"
            try:
                with zipfile.ZipFile(io.BytesIO(file_bytes)) as zf:
                    with zf.open("word/document.xml") as f:
                        root = ET.parse(f).getroot()
                paragraphs = []
                for para in root.iter(f"{{{ns}}}p"):
                    text = "".join(t.text or "" for t in para.iter(f"{{{ns}}}t"))
                    if text.strip():
                        paragraphs.append(text)
                return "\n".join(paragraphs)
            except Exception:
                return ""
        # .txt, .md, .csv — all decoded as UTF-8 plain text
        return file_bytes.decode("utf-8", errors="replace")

    async def ingest(self, file_bytes: bytes, filename: str, collection_name: str = "") -> dict:
        """Parse, chunk (parent 1500/200, child 400/50), embed, and upsert to Qdrant."""
        cname = collection_name or settings.QDRANT_COLLECTION
        text = self._parse_text(file_bytes, filename)
        if not text.strip():
            logger.warning("ingestion_empty_document", filename=filename)
            return {"filename": filename, "parent_count": 0, "child_count": 0, "status": "empty"}

        parent_chunks = split_text(text, settings.PARENT_CHUNK_SIZE, 200)

        children: List[dict] = []
        for parent_text in parent_chunks:
            parent_id = str(uuid.uuid4())
            for idx, child_text in enumerate(
                split_text(parent_text, settings.CHILD_CHUNK_SIZE, 50)
            ):
                children.append(
                    {
                        "text": child_text,
                        "metadata": {
                            "parent_id": parent_id,
                            "parent_text": parent_text,
                            "filename": filename,
                            "chunk_index": idx,
                        },
                    }
                )

        self._ensure_collection(cname)
        client = get_qdrant_client()
        embeddings = get_embeddings()
        try:
            for batch_start in range(0, len(children), EMBED_BATCH_SIZE):
                batch = children[batch_start : batch_start + EMBED_BATCH_SIZE]
                # HuggingFaceEmbeddings is sync — run in executor to not block the event loop.
                texts = [c["text"] for c in batch]
                vectors = await asyncio.get_event_loop().run_in_executor(
                    None, embeddings.embed_documents, texts
                )
                points = [
                    PointStruct(
                        id=str(uuid.uuid4()),
                        vector=vector,
                        payload={"child_text": child["text"], **child["metadata"]},
                    )
                    for child, vector in zip(batch, vectors)
                ]
                client.upsert(collection_name=cname, points=points)
        except Exception as exc:
            logger.error("ingestion_upsert_failed", filename=filename, error=str(exc))
            raise

        logger.info(
            "document_ingested",
            filename=filename,
            parent_count=len(parent_chunks),
            child_count=len(children),
        )
        return {
            "filename": filename,
            "parent_count": len(parent_chunks),
            "child_count": len(children),
            "status": "indexed",
        }

    async def retrieve(self, query: str, top_k: int = 5) -> List[str]:
        """Search child vectors and return deduplicated parent_text contexts."""
        embeddings = get_embeddings()
        try:
            query_vector = await asyncio.get_event_loop().run_in_executor(
                None, embeddings.embed_query, query
            )
            client = get_qdrant_client()
            # qdrant-client >= 1.16 removed QdrantClient.search.
            results = client.query_points(
                collection_name=settings.QDRANT_COLLECTION,
                query=query_vector,
                limit=top_k * 3,
            ).points
        except Exception as exc:
            logger.error("retrieval_failed", query=query, error=str(exc))
            return []

        seen_parents: set = set()
        contexts: List[str] = []
        for hit in results:
            payload = hit.payload or {}
            parent_id = payload.get("parent_id")
            if parent_id in seen_parents:
                continue
            seen_parents.add(parent_id)
            contexts.append(payload.get("parent_text", ""))
            if len(contexts) >= top_k:
                break
        return contexts
