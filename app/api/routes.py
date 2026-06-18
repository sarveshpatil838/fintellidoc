"""
FastAPI route definitions for FintelliDoc API.

All endpoints follow the same pattern:
1. Validate request (FastAPI + Pydantic handles this automatically)
2. Call the appropriate service
3. Return structured response
4. On error: log + return appropriate HTTP error (never 500 with stack trace)

Dependency injection is used for services so they can be swapped in tests.
"""

import uuid
from fastapi import APIRouter, Depends, HTTPException, status
from fastapi.responses import JSONResponse

from app.models.schemas import (
    ExtractionRequest,
    ExtractionResponse,
    QueryRequest,
    QueryResponse,
    RiskRequest,
    RiskResponse,
    CompareRequest,
    CompareResponse,
    IndexRequest,
    IndexResponse,
    HealthResponse,
)
from app.services.extraction import ExtractionService
from app.services.rag import RAGService
from app.services.risk_detection import RiskDetectionService
from app.services.compare import CompareService
from app.services.validation import LLMValidationError
from app.core.config import get_settings
from app.core.logging import get_logger

logger = get_logger("routes")
settings = get_settings()
router = APIRouter()

# ─── Service singletons (initialized once at startup) ─────────────────────────
_extraction_service: ExtractionService | None = None
_rag_service: RAGService | None = None
_risk_service: RiskDetectionService | None = None
_compare_service: CompareService | None = None


def get_extraction_service() -> ExtractionService:
    global _extraction_service
    if _extraction_service is None:
        _extraction_service = ExtractionService()
    return _extraction_service


def get_rag_service() -> RAGService:
    global _rag_service
    if _rag_service is None:
        _rag_service = RAGService()
    return _rag_service


def get_risk_service() -> RiskDetectionService:
    global _risk_service
    if _risk_service is None:
        _risk_service = RiskDetectionService()
    return _risk_service


def get_compare_service() -> CompareService:
    global _compare_service
    if _compare_service is None:
        _compare_service = CompareService()
    return _compare_service


# ─── Endpoints ────────────────────────────────────────────────────────────────

@router.get("/health", response_model=HealthResponse, tags=["System"])
async def health_check():
    """Health check endpoint. Returns model and version info."""
    return HealthResponse(model=settings.claude_model)


@router.post(
    "/extract",
    response_model=ExtractionResponse,
    tags=["Extraction"],
    summary="Extract structured data from a financial document",
    description=(
        "Extracts entities, financial metrics, risk signals, and key dates from "
        "unstructured financial text using Claude's tool use API. "
        "Includes automatic retry and validation — never returns corrupt data."
    )
)
async def extract(
    request: ExtractionRequest,
    service: ExtractionService = Depends(get_extraction_service),
):
    # Assign a doc_id if not provided
    if not request.doc_id:
        request.doc_id = str(uuid.uuid4())

    try:
        result = await service.extract(request)
        logger.info("extract_endpoint_success", doc_id=request.doc_id)
        return result
    except LLMValidationError as e:
        logger.error("extract_validation_failed", doc_id=request.doc_id, error=str(e))
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail=f"AI extraction failed validation after {e.attempts} attempts: {str(e.last_error)}"
        )
    except Exception as e:
        logger.error("extract_unexpected_error", doc_id=request.doc_id, error=str(e))
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Extraction failed. See logs for details."
        )


@router.post(
    "/index",
    response_model=IndexResponse,
    tags=["RAG"],
    summary="Index a document for semantic search",
)
async def index_document(
    request: IndexRequest,
    service: RAGService = Depends(get_rag_service),
):
    doc_id = request.doc_id or str(uuid.uuid4())
    try:
        chunks = await service.index_document(request.text, doc_id, request.metadata)
        return IndexResponse(doc_id=doc_id, chunks_indexed=chunks)
    except Exception as e:
        logger.error("index_failed", doc_id=doc_id, error=str(e))
        raise HTTPException(status_code=500, detail="Indexing failed.")


@router.post(
    "/query",
    response_model=QueryResponse,
    tags=["RAG"],
    summary="Ask a natural language question against indexed documents",
)
async def query(
    request: QueryRequest,
    service: RAGService = Depends(get_rag_service),
):
    try:
        result = await service.query(request.question, request.doc_ids)
        return QueryResponse(
            question=request.question,
            answer=result["answer"],
            sources=result["sources"],
            confidence=result["confidence"],
            processing_time_ms=result["processing_time_ms"],
        )
    except Exception as e:
        logger.error("query_failed", question=request.question[:100], error=str(e))
        raise HTTPException(status_code=500, detail="Query failed.")


@router.post(
    "/risks",
    response_model=RiskResponse,
    tags=["Risk"],
    summary="Detect material risk signals in a financial document",
)
async def detect_risks(
    request: RiskRequest,
    service: RiskDetectionService = Depends(get_risk_service),
):
    try:
        return await service.detect_risks(request)
    except LLMValidationError as e:
        raise HTTPException(status_code=422, detail=str(e))
    except Exception as e:
        logger.error("risk_detection_failed", error=str(e))
        raise HTTPException(status_code=500, detail="Risk detection failed.")


@router.post(
    "/compare",
    response_model=CompareResponse,
    tags=["Analysis"],
    summary="Compare two financial documents and surface what changed",
)
async def compare_documents(
    request: CompareRequest,
    service: CompareService = Depends(get_compare_service),
):
    try:
        return await service.compare(request)
    except LLMValidationError as e:
        raise HTTPException(status_code=422, detail=str(e))
    except Exception as e:
        logger.error("compare_failed", error=str(e))
        raise HTTPException(status_code=500, detail="Comparison failed.")
