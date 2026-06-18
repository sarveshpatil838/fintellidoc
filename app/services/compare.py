"""
Document comparison service — diff two financial documents.

Use case: Compare Q3 2024 earnings vs Q3 2023 to surface what changed.
Or compare two versions of a contract to flag material modifications.

This is genuinely hard for humans at scale and genuinely valuable for
investment analysts who need to track disclosures across periods.
"""

import time
import asyncio
import anthropic
from tenacity import retry, stop_after_attempt, wait_exponential, retry_if_exception_type

from app.core.config import get_settings
from app.core.logging import get_logger
from app.models.schemas import CompareRequest, CompareResponse, LLMCompareOutput
from app.services.validation import validate_llm_output

logger = get_logger("compare")
settings = get_settings()

COMPARE_TOOL = {
    "name": "compare_documents",
    "description": "Identify what changed between two versions of a financial document.",
    "input_schema": {
        "type": "object",
        "properties": {
            "changes": {
                "type": "array",
                "items": {
                    "type": "object",
                    "properties": {
                        "change_type": {"type": "string", "enum": ["added", "removed", "modified"]},
                        "section": {"type": "string"},
                        "description": {"type": "string"},
                        "significance": {"type": "string", "enum": ["material", "minor", "formatting"]}
                    },
                    "required": ["change_type", "section", "description", "significance"]
                }
            },
            "summary": {"type": "string"}
        },
        "required": ["changes", "summary"]
    }
}


class CompareService:
    def __init__(self):
        self.client = anthropic.Anthropic(api_key=settings.anthropic_api_key)

    @retry(
        stop=stop_after_attempt(settings.max_retries),
        wait=wait_exponential(min=1, max=8),
        retry=retry_if_exception_type((anthropic.APIError, anthropic.RateLimitError)),
        reraise=True,
    )
    def _call_claude(self, text_a: str, text_b: str, focus: str | None) -> anthropic.types.Message:
        focus_instruction = f"\nPay special attention to changes in: {focus}" if focus else ""
        return self.client.messages.create(
            model=settings.claude_model,
            max_tokens=2048,
            tools=[COMPARE_TOOL],
            tool_choice={"type": "tool", "name": "compare_documents"},
            messages=[{
                "role": "user",
                "content": (
                    f"Compare these two financial documents and identify what changed.{focus_instruction}\n\n"
                    f"DOCUMENT A (older):\n{text_a}\n\n"
                    f"DOCUMENT B (newer):\n{text_b}"
                )
            }]
        )

    async def compare(self, request: CompareRequest) -> CompareResponse:
        start_time = time.time()

        response = await asyncio.get_event_loop().run_in_executor(
            None, lambda: self._call_claude(request.text_a, request.text_b, request.focus)
        )

        tool_use_block = next((b for b in response.content if b.type == "tool_use"), None)
        if not tool_use_block:
            raise ValueError("No tool_use block in compare response")

        validated, _ = validate_llm_output(tool_use_block.input, LLMCompareOutput, "compare")
        material_count = sum(1 for c in validated.changes if c.significance == "material")

        return CompareResponse(
            changes=validated.changes,
            material_changes_count=material_count,
            summary=validated.summary,
            processing_time_ms=int((time.time() - start_time) * 1000),
        )
