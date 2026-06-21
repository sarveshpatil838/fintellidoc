"""
Tests for the extraction service.

Tests are organized into:
1. Schema validation tests (no API calls — fast)
2. Validation layer tests (no API calls — fast)
3. Integration tests (require ANTHROPIC_API_KEY — marked with @pytest.mark.integration)
"""

import pytest
import json
from pydantic import ValidationError

from app.models.schemas import (
    ExtractionRequest,
    LLMExtractionOutput,
    Entity,
    FinancialMetric,
    EntityType,
    DocType,
)
from app.services.validation import validate_llm_output, LLMValidationError


# ─── Schema Validation Tests ──────────────────────────────────────────────────

class TestExtractionRequest:
    def test_valid_request(self):
        req = ExtractionRequest(
            text="Apple Inc. reported Q3 2024 revenue of $85.8 billion, up 5% year-over-year.",
            doc_type=DocType.EARNINGS,
        )
        assert req.doc_type == DocType.EARNINGS
        assert len(req.text) > 0

    def test_text_too_short(self):
        with pytest.raises(ValidationError):
            ExtractionRequest(text="too short")

    def test_whitespace_only_text(self):
        with pytest.raises(ValidationError):
            ExtractionRequest(text="   " * 20)

    def test_default_doc_type_is_unknown(self):
        req = ExtractionRequest(text="A" * 100)
        assert req.doc_type == DocType.UNKNOWN


class TestEntity:
    def test_valid_entity(self):
        e = Entity(name="Apple Inc.", entity_type=EntityType.COMPANY, confidence=0.98)
        assert e.name == "Apple Inc."
        assert e.confidence == 0.98

    def test_confidence_out_of_range(self):
        with pytest.raises(ValidationError):
            Entity(name="Apple", entity_type=EntityType.COMPANY, confidence=1.5)

    def test_confidence_negative(self):
        with pytest.raises(ValidationError):
            Entity(name="Apple", entity_type=EntityType.COMPANY, confidence=-0.1)


# ─── Validation Layer Tests ───────────────────────────────────────────────────

class TestValidationLayer:
    def test_valid_output_passes(self):
        valid_data = {
            "entities": [
                {"name": "Apple Inc.", "entity_type": "company", "confidence": 0.98}
            ],
            "financials": [
                {"metric": "revenue", "value": 85.8, "value_text": "$85.8 billion", "period": "Q3 2024"}
            ],
            "risks": [],
            "key_dates": ["Q3 2024"],
            "summary": "Apple reported strong Q3 results.",
            "doc_type_detected": "earnings",
        }
        result, retries = validate_llm_output(valid_data, LLMExtractionOutput, "test")
        assert isinstance(result, LLMExtractionOutput)
        assert retries == 0
        assert len(result.entities) == 1
        assert result.entities[0].name == "Apple Inc."

    def test_invalid_output_raises(self):
        # Missing required fields
        invalid_data = {
            "entities": [{"name": "Apple"}],  # missing entity_type and confidence
        }
        with pytest.raises(LLMValidationError):
            validate_llm_output(invalid_data, LLMExtractionOutput, "test")

    def test_malformed_json_string_raises(self):
        with pytest.raises(LLMValidationError):
            validate_llm_output("not valid json {{{", LLMExtractionOutput, "test")

    def test_valid_json_string_passes(self):
        valid_data = {
            "entities": [],
            "financials": [],
            "risks": [],
            "key_dates": [],
            "summary": "Empty document.",
            "doc_type_detected": "unknown",
        }
        json_str = json.dumps(valid_data)
        result, retries = validate_llm_output(json_str, LLMExtractionOutput, "test")
        assert isinstance(result, LLMExtractionOutput)

    def test_wrong_confidence_type_raises(self):
        data = {
            "entities": [
                {"name": "Apple", "entity_type": "company", "confidence": "high"}  # should be float
            ],
            "financials": [],
            "risks": [],
            "key_dates": [],
            "summary": "test",
            "doc_type_detected": "unknown",
        }
        with pytest.raises(LLMValidationError):
            validate_llm_output(data, LLMExtractionOutput, "test")


# ─── Integration Tests ────────────────────────────────────────────────────────

@pytest.mark.integration
@pytest.mark.asyncio
async def test_real_extraction():
    """
    Integration test — requires ANTHROPIC_API_KEY in environment.
    Run with: pytest tests/ -v -m integration
    """
    import os
    if not os.getenv("ANTHROPIC_API_KEY"):
        pytest.skip("ANTHROPIC_API_KEY not set")

    from app.services.extraction import ExtractionService

    service = ExtractionService()
    request = ExtractionRequest(
        text=(
            "Apple Inc. (AAPL) today announced financial results for its fiscal 2024 "
            "third quarter ended June 29, 2024. The Company posted quarterly revenue of "
            "$85.8 billion, up 5 percent year over year, and quarterly earnings per "
            "diluted share of $1.40, up 11 percent year over year. "
            "We are happy to report that we had an all-time revenue record in Services and a June quarter record for iPhone, said Tim Cook."
        ),
        doc_type=DocType.EARNINGS,
    )

    result = await service.extract(request)
    assert result.doc_type == DocType.EARNINGS
    assert len(result.entities) > 0
    assert any(e.name == "Apple Inc." or "Apple" in e.name for e in result.entities)
    assert result.processing_time_ms > 0
    assert result.model == "claude-sonnet-4-6"
