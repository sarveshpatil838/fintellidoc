"""
Risk signal detection service.

Financial documents contain language that signals material risks —
going-concern qualifications, covenant violations, litigation exposure,
regulatory investigations. This service surfaces those signals explicitly.

Why a separate service? Risk detection requires different prompting than
general extraction. We want high recall (better to flag too much than miss
a going-concern warning) and precise excerpts for human review.
"""

import time
import asyncio
import json
import anthropic
from tenacity import retry, stop_after_attempt, wait_exponential, retry_if_exception_type

from app.core.config import get_settings
from app.core.logging import get_logger
from app.models.schemas import RiskRequest, RiskResponse, LLMRiskOutput, RiskSeverity
from app.services.validation import validate_llm_output, LLMValidationError

logger = get_logger("risk_detection")
settings = get_settings()

RISK_TOOL = {
    "name": "detect_risk_signals",
    "description": "Identify and classify financial risk signals in the document.",
    "input_schema": {
        "type": "object",
        "properties": {
            "risk_signals": {
                "type": "array",
                "items": {
                    "type": "object",
                    "properties": {
                        "signal_type": {
                            "type": "string",
                            "enum": [
                                "going_concern",
                                "covenant_violation",
                                "material_uncertainty",
                                "litigation",
                                "regulatory_investigation",
                                "revenue_decline",
                                "debt_maturity",
                                "customer_concentration",
                                "liquidity_risk",
                                "other"
                            ]
                        },
                        "severity": {"type": "string", "enum": ["high", "medium", "low"]},
                        "excerpt": {"type": "string", "description": "Exact quote from document"},
                        "explanation": {"type": "string"}
                    },
                    "required": ["signal_type", "severity", "excerpt", "explanation"]
                }
            },
            "overall_risk_level": {"type": "string", "enum": ["high", "medium", "low"]},
            "summary": {"type": "string"}
        },
        "required": ["risk_signals", "overall_risk_level", "summary"]
    }
}

RISK_SYSTEM_PROMPT = """You are a financial risk analyst specializing in identifying 
material risk disclosures in financial documents.

Your task is to identify ALL risk signals with HIGH RECALL — it is better to flag 
a potential risk than to miss it. Human reviewers will make final judgments.

Focus especially on:
- Going-concern language ("substantial doubt", "ability to continue as a going concern")
- Covenant violations or waivers
- Material uncertainties
- Significant litigation or regulatory investigations
- Unusual revenue declines or customer losses
- Approaching debt maturities with unclear refinancing plans
- Liquidity concerns

Always include the exact excerpt from the document to support each signal."""


class RiskDetectionService:
    def __init__(self):
        self.client = anthropic.Anthropic(api_key=settings.anthropic_api_key)

    @retry(
        stop=stop_after_attempt(settings.max_retries),
        wait=wait_exponential(min=1, max=8),
        retry=retry_if_exception_type((anthropic.APIError, anthropic.RateLimitError)),
        reraise=True,
    )
    def _call_claude(self, text: str) -> anthropic.types.Message:
        return self.client.messages.create(
            model=settings.claude_model,
            max_tokens=2048,
            system=RISK_SYSTEM_PROMPT,
            tools=[RISK_TOOL],
            tool_choice={"type": "tool", "name": "detect_risk_signals"},
            messages=[{
                "role": "user",
                "content": f"Identify all risk signals in the following financial document:\n\n{text}"
            }]
        )

    async def detect_risks(self, request: RiskRequest) -> RiskResponse:
        start_time = time.time()

        response = await asyncio.get_event_loop().run_in_executor(
            None, lambda: self._call_claude(request.text)
        )

        tool_use_block = next(
            (b for b in response.content if b.type == "tool_use"), None
        )
        if not tool_use_block:
            raise ValueError("No tool_use block in risk detection response")

        validated, retry_count = validate_llm_output(
            tool_use_block.input,
            LLMRiskOutput,
            context="risk_detection"
        )

        processing_time_ms = int((time.time() - start_time) * 1000)

        logger.info(
            "risk_detection_completed",
            signals_found=len(validated.risk_signals),
            overall_risk=validated.overall_risk_level,
            processing_time_ms=processing_time_ms,
        )

        return RiskResponse(
            risk_signals=validated.risk_signals,
            overall_risk_level=validated.overall_risk_level,
            summary=validated.summary,
            processing_time_ms=processing_time_ms,
        )
