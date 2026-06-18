"""
Pydantic v2 schemas for all request/response models.

These schemas serve dual purpose:
1. FastAPI request/response validation (automatic 422 on bad input)
2. LLM output validation — Claude's responses are parsed against these
   schemas. If the model returns malformed output, validation raises
   ValidationError which triggers the retry layer.
"""

from pydantic import BaseModel, Field, field_validator
from typing import Optional, Literal
from enum import Enum
import uuid


# ─── Enums ────────────────────────────────────────────────────────────────────

class DocType(str, Enum):
    EARNINGS = "earnings"
    FILING_10K = "10k"
    FILING_10Q = "10q"
    CONTRACT = "contract"
    RESEARCH_REPORT = "research_report"
    PRESS_RELEASE = "press_release"
    UNKNOWN = "unknown"


class EntityType(str, Enum):
    COMPANY = "company"
    PERSON = "person"
    FINANCIAL_METRIC = "financial_metric"
    DATE = "date"
    LOCATION = "location"
    PRODUCT = "product"
    RISK_FACTOR = "risk_factor"


class RiskSeverity(str, Enum):
    HIGH = "high"
    MEDIUM = "medium"
    LOW = "low"


# ─── Sub-models ───────────────────────────────────────────────────────────────

class Entity(BaseModel):
    name: str = Field(..., min_length=1)
    entity_type: EntityType
    confidence: float = Field(..., ge=0.0, le=1.0)
    context: Optional[str] = None


class FinancialMetric(BaseModel):
    metric: str = Field(..., description="e.g. revenue, net_income, eps")
    value: Optional[float] = None
    value_text: str = Field(..., description="Raw text value as it appeared")
    unit: Optional[str] = None
    period: Optional[str] = None
    yoy_change: Optional[str] = None


class RiskSignal(BaseModel):
    signal_type: str = Field(..., description="e.g. going_concern, covenant_violation, litigation")
    severity: RiskSeverity
    excerpt: str = Field(..., description="Exact quote from the document")
    explanation: str


class DocumentChange(BaseModel):
    change_type: Literal["added", "removed", "modified"]
    section: str
    description: str
    significance: Literal["material", "minor", "formatting"]


# ─── Request Models ───────────────────────────────────────────────────────────

class ExtractionRequest(BaseModel):
    text: str = Field(..., min_length=50, max_length=500_000)
    doc_type: DocType = DocType.UNKNOWN
    extract_fields: list[Literal["entities", "financials", "risks", "dates"]] = Field(
        default=["entities", "financials", "risks"]
    )
    doc_id: Optional[str] = None

    @field_validator("text")
    @classmethod
    def text_not_whitespace(cls, v: str) -> str:
        if not v.strip():
            raise ValueError("text cannot be empty or whitespace")
        return v.strip()


class QueryRequest(BaseModel):
    question: str = Field(..., min_length=5, max_length=1000)
    doc_ids: Optional[list[str]] = None  # None = search all indexed docs
    top_k: int = Field(default=5, ge=1, le=20)


class RiskRequest(BaseModel):
    text: str = Field(..., min_length=50)
    doc_type: DocType = DocType.UNKNOWN


class CompareRequest(BaseModel):
    text_a: str = Field(..., min_length=50, description="Older document")
    text_b: str = Field(..., min_length=50, description="Newer document")
    focus: Optional[str] = Field(None, description="Optional focus area, e.g. 'revenue guidance'")


class IndexRequest(BaseModel):
    text: str = Field(..., min_length=50)
    doc_id: Optional[str] = None
    metadata: dict = Field(default_factory=dict)


# ─── Response Models ──────────────────────────────────────────────────────────

class ExtractionResponse(BaseModel):
    doc_id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    doc_type: DocType
    entities: list[Entity] = []
    financials: list[FinancialMetric] = []
    risks: list[RiskSignal] = []
    key_dates: list[str] = []
    summary: Optional[str] = None
    processing_time_ms: int
    model: str
    retry_count: int = 0


class QueryResponse(BaseModel):
    question: str
    answer: str
    sources: list[dict] = []
    confidence: float = Field(..., ge=0.0, le=1.0)
    processing_time_ms: int


class RiskResponse(BaseModel):
    doc_id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    risk_signals: list[RiskSignal] = []
    overall_risk_level: RiskSeverity
    summary: str
    processing_time_ms: int


class CompareResponse(BaseModel):
    changes: list[DocumentChange] = []
    material_changes_count: int
    summary: str
    processing_time_ms: int


class IndexResponse(BaseModel):
    doc_id: str
    chunks_indexed: int
    status: str = "indexed"


class HealthResponse(BaseModel):
    status: str = "ok"
    model: str
    version: str = "1.0.0"


# ─── Internal models (LLM output contracts) ───────────────────────────────────

class LLMExtractionOutput(BaseModel):
    """
    This is what we expect Claude to return via tool use.
    Strict validation — missing fields = retry.
    """
    entities: list[Entity] = []
    financials: list[FinancialMetric] = []
    risks: list[RiskSignal] = []
    key_dates: list[str] = []
    summary: str = ""
    doc_type_detected: DocType = DocType.UNKNOWN


class LLMRiskOutput(BaseModel):
    risk_signals: list[RiskSignal] = []
    overall_risk_level: RiskSeverity = RiskSeverity.LOW
    summary: str


class LLMCompareOutput(BaseModel):
    changes: list[DocumentChange] = []
    summary: str
