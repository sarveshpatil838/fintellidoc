"""
Document extraction service using Anthropic Claude API with tool use.

Key design decisions:
1. Tool use (function calling) over free-text JSON — more reliable structured output
2. Retry at the API call level (tenacity) AND at the validation level
3. All prompts are versioned constants — easy to A/B test and audit
4. Processing time is always measured and returned — for SLO monitoring
5. The model version is always included in the response — for debugging regressions
"""

import time
import json
import asyncio
from typing import Optional

import anthropic
from tenacity import retry, stop_after_attempt, wait_exponential, retry_if_exception_type

from app.core.config import get_settings
from app.core.logging import get_logger
from app.models.schemas import (
    ExtractionRequest,
    ExtractionResponse,
    LLMExtractionOutput,
    DocType,
    Entity,
    FinancialMetric,
    RiskSignal,
    EntityType,
    RiskSeverity,
)
from app.services.validation import validate_llm_output, LLMValidationError

logger = get_logger("extraction")
settings = get_settings()

# ─── Tool Definition ──────────────────────────────────────────────────────────
# We define the extraction schema as a Claude tool.
# This forces the model to return structured data rather than free-form text.
# Far more reliable than "return JSON with these fields".

EXTRACTION_TOOL = {
    "name": "extract_financial_document",
    "description": (
        "Extract structured information from a financial document. "
        "Call this tool with all extracted entities, financial metrics, "
        "risk signals, key dates, and a brief summary."
    ),
    "input_schema": {
        "type": "object",
        "properties": {
            "entities": {
                "type": "array",
                "description": "Named entities in the document",
                "items": {
                    "type": "object",
                    "properties": {
                        "name": {"type": "string"},
                        "entity_type": {
                            "type": "string",
                            "enum": ["company", "person", "financial_metric", "date", "location", "product", "risk_factor"]
                        },
                        "confidence": {"type": "number", "minimum": 0, "maximum": 1},
                        "context": {"type": "string"}
                    },
                    "required": ["name", "entity_type", "confidence"]
                }
            },
            "financials": {
                "type": "array",
                "description": "Financial metrics and figures",
                "items": {
                    "type": "object",
                    "properties": {
                        "metric": {"type": "string"},
                        "value": {"type": "number"},
                        "value_text": {"type": "string"},
                        "unit": {"type": "string"},
                        "period": {"type": "string"},
                        "yoy_change": {"type": "string"}
                    },
                    "required": ["metric", "value_text"]
                }
            },
            "risks": {
                "type": "array",
                "description": "Risk signals and warnings",
                "items": {
                    "type": "object",
                    "properties": {
                        "signal_type": {"type": "string"},
                        "severity": {"type": "string", "enum": ["high", "medium", "low"]},
                        "excerpt": {"type": "string"},
                        "explanation": {"type": "string"}
                    },
                    "required": ["signal_type", "severity", "excerpt", "explanation"]
                }
            },
            "key_dates": {
                "type": "array",
                "items": {"type": "string"},
                "description": "Important dates mentioned"
            },
            "summary": {
                "type": "string",
                "description": "2-3 sentence summary of the document"
            },
            "doc_type_detected": {
                "type": "string",
                "enum": ["earnings", "10k", "10q", "contract", "research_report", "press_release", "unknown"]
            }
        },
        "required": ["entities", "financials", "risks", "key_dates", "summary", "doc_type_detected"]
    }
}

# ─── Prompt Templates ─────────────────────────────────────────────────────────

SYSTEM_PROMPT = """You are a financial document analyst with deep expertise in SEC filings, 
earnings reports, analyst research, and financial contracts.

Your task is to extract structured information from financial documents with high accuracy.

Guidelines:
- Extract ALL named entities (companies, executives, products, locations)
- Capture financial metrics with their exact values and periods
- Flag risk language precisely — going-concern, covenant violations, material uncertainties
- Confidence scores should reflect actual certainty (not always 1.0)
- Be conservative: if something is ambiguous, reflect that in the confidence score
- ALWAYS call the extract_financial_document tool with your findings"""


def build_extraction_prompt(text: str, doc_type: DocType) -> str:
    doc_hint = f" This appears to be a {doc_type.value} document." if doc_type != DocType.UNKNOWN else ""
    return f"Please extract all relevant information from the following financial document.{doc_hint}\n\n---\n\n{text}"


# ─── Service ──────────────────────────────────────────────────────────────────

class ExtractionService:
    def __init__(self):
        self.client = anthropic.Anthropic(api_key=settings.anthropic_api_key)
        self.model = settings.claude_model

    @retry(
        stop=stop_after_attempt(settings.max_retries),
        wait=wait_exponential(multiplier=1, min=1, max=8),
        retry=retry_if_exception_type((anthropic.APIError, anthropic.RateLimitError)),
        reraise=True,
    )
    def _call_claude(self, text: str, doc_type: DocType) -> anthropic.types.Message:
        """
        Make the API call to Claude with tool use.
        Retried automatically on API errors (rate limits, server errors).
        Does NOT retry on validation failures — that's handled separately.
        """
        return self.client.messages.create(
            model=self.model,
            max_tokens=settings.max_tokens,
            system=SYSTEM_PROMPT,
            tools=[EXTRACTION_TOOL],
            tool_choice={"type": "tool", "name": "extract_financial_document"},
            messages=[{
                "role": "user",
                "content": build_extraction_prompt(text, doc_type)
            }]
        )

    async def extract(self, request: ExtractionRequest) -> ExtractionResponse:
        """
        Main extraction pipeline:
        1. Call Claude with tool use (retry on API errors)
        2. Parse tool use result
        3. Validate against Pydantic schema (retry on validation errors)
        4. Return structured response

        Never returns partial/corrupt data — raises explicitly on failure.
        """
        start_time = time.time()
        retry_count = 0

        logger.info(
            "extraction_started",
            doc_type=request.doc_type.value,
            text_length=len(request.text),
            doc_id=request.doc_id,
        )

        # Retry loop for validation failures (separate from API retry above)
        for attempt in range(settings.max_retries):
            try:
                # Run blocking API call in thread pool
                response = await asyncio.get_event_loop().run_in_executor(
                    None,
                    lambda: self._call_claude(request.text, request.doc_type)
                )

                # Find the tool_use block in the response
                tool_use_block = next(
                    (b for b in response.content if b.type == "tool_use"),
                    None
                )
                if not tool_use_block:
                    raise ValueError(f"No tool_use block in response. Stop reason: {response.stop_reason}")

                # Validate the tool use input against our schema
                raw_data = tool_use_block.input
                validated_output, val_retries = validate_llm_output(
                    raw_data,
                    LLMExtractionOutput,
                    context=f"extraction attempt {attempt + 1}"
                )
                retry_count = attempt + val_retries

                processing_time_ms = int((time.time() - start_time) * 1000)

                logger.info(
                    "extraction_completed",
                    doc_id=request.doc_id,
                    entities_found=len(validated_output.entities),
                    financials_found=len(validated_output.financials),
                    risks_found=len(validated_output.risks),
                    processing_time_ms=processing_time_ms,
                    retry_count=retry_count,
                )

                return ExtractionResponse(
                    doc_id=request.doc_id or "",
                    doc_type=validated_output.doc_type_detected,
                    entities=validated_output.entities,
                    financials=validated_output.financials,
                    risks=validated_output.risks,
                    key_dates=validated_output.key_dates,
                    summary=validated_output.summary,
                    processing_time_ms=processing_time_ms,
                    model=self.model,
                    retry_count=retry_count,
                )

            except LLMValidationError as e:
                retry_count += e.attempts
                if attempt == settings.max_retries - 1:
                    logger.error(
                        "extraction_failed_all_retries",
                        doc_id=request.doc_id,
                        attempts=retry_count,
                        error=str(e),
                    )
                    raise
                logger.warning(
                    "extraction_retrying",
                    attempt=attempt + 1,
                    error=str(e),
                )

        # Should not reach here
        raise RuntimeError("Extraction failed: exhausted all retry attempts")
