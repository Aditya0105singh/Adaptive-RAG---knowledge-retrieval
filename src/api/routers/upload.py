"""Document upload and session-history endpoints."""
from datetime import datetime
from typing import List

from fastapi import APIRouter, Depends, Form, HTTPException, UploadFile

from src.api.middleware.auth import verify_api_key
from src.api.schemas import SessionMessage, UploadResponse
from src.core.database import get_chat_collection
from src.core.logging import get_logger
from src.services.ingestion import DocumentIngestionService

logger = get_logger(__name__)
router = APIRouter(prefix="/api", tags=["upload"])

MAX_FILE_SIZE = 50 * 1024 * 1024  # 50 MB
ALLOWED_EXTENSIONS = (".pdf", ".txt", ".docx", ".md", ".csv")

_ingestion_service = DocumentIngestionService()


@router.post("/upload", response_model=UploadResponse, dependencies=[Depends(verify_api_key)])
async def upload_document(
    file: UploadFile,
    session_id: str = Form(default="default"),
) -> UploadResponse:
    """Validate and ingest a PDF/TXT file into the session-scoped Qdrant collection."""
    filename = file.filename or "unknown"
    if not filename.lower().endswith(ALLOWED_EXTENSIONS):
        raise HTTPException(
            status_code=422,
            detail={"error": "invalid_file_type", "message": "Only PDF, TXT, DOCX, MD, and CSV files are supported."},
        )

    file_bytes = await file.read()
    if len(file_bytes) > MAX_FILE_SIZE:
        raise HTTPException(
            status_code=413,
            detail={"error": "file_too_large", "message": "Maximum file size is 50MB."},
        )

    collection_name = f"session_{session_id}"
    try:
        result = await _ingestion_service.ingest(file_bytes, filename, collection_name)
    except Exception as exc:
        logger.error("upload_ingestion_failed", filename=filename, error=str(exc))
        raise HTTPException(
            status_code=500,
            detail={"error": "ingestion_failed", "message": str(exc)},
        )
    return UploadResponse(**result)


@router.get("/sessions/{session_id}", response_model=List[SessionMessage])
async def get_session_history(session_id: str) -> List[SessionMessage]:
    """Return the last 20 chat messages for a session, oldest first."""
    try:
        collection = get_chat_collection()
        if collection is None:
            return []  # MongoDB not configured — return empty history
        cursor = (
            collection.find({"session_id": session_id})
            .sort("timestamp", -1)
            .limit(20)
        )
        messages = list(cursor)[::-1]
    except Exception as exc:
        logger.error("session_history_fetch_failed", session_id=session_id, error=str(exc))
        return []  # Don't crash — just return empty history
    return [
        SessionMessage(
            role=msg.get("role", "user"),
            content=msg.get("content", ""),
            timestamp=msg.get("timestamp", datetime.utcnow()),
        )
        for msg in messages
    ]

@router.get("/sessions/{session_id}/cost")
async def get_session_cost(session_id: str) -> dict:
    """Calculate the total estimated cost for a session."""
    try:
        collection = get_chat_collection()
        if collection is None:
            return {"total_cost_usd": 0.0}
        
        pipeline = [
            {"$match": {"session_id": session_id, "role": "assistant"}},
            {"$group": {"_id": None, "total_cost": {"$sum": "$metadata.estimated_cost_usd"}}}
        ]
        result = list(collection.aggregate(pipeline))
        total = result[0]["total_cost"] if result else 0.0
        return {"total_cost_usd": round(total, 6)}
    except Exception as exc:
        logger.error("session_cost_fetch_failed", session_id=session_id, error=str(exc))
        return {"total_cost_usd": 0.0}


@router.get("/sessions/{session_id}/last-metadata")
async def get_last_metadata(session_id: str) -> dict:
    """Return the raw metadata JSON of the last assistant response for the Pipeline Inspector."""
    try:
        collection = get_chat_collection()
        if collection is None:
            return {"status": "MongoDB not configured"}
        
        last_msg = collection.find_one(
            {"session_id": session_id, "role": "assistant"},
            sort=[("timestamp", -1)]
        )
        if not last_msg or "metadata" not in last_msg:
            return {"status": "No requests found for this session yet."}
            
        return last_msg["metadata"]
    except Exception as exc:
        logger.error("session_metadata_fetch_failed", session_id=session_id, error=str(exc))
        return {"error": "Failed to fetch metadata."}

