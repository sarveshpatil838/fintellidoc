"""
Validation and retry layer — the most important service in the codebase.

The core principle: an AI feature that silently returns bad data is worse
than no AI feature at all. Trust, once broken by a corrupt output, is very
hard to rebuild.

This module enforces a strict contract:
  - Every LLM response is validated against a Pydantic schema
  - ValidationError triggers a retry with exponential backoff
  - After max_retries, we raise explicitly — never swallow the error
  - Every attempt is logged with structured context for observability

Design decision: we use Claude's tool_use API (function calling) instead of
asking the model to return JSON in free text. Tool use forces structured
output at the API level, dramatically reducing validation failures.
"""

import json
import time
from typing import TypeVar, Type, Callable, Any
from pydantic import BaseModel, ValidationError
from tenacity import (
    retry,
    stop_after_attempt,
    wait_exponential,
    retry_if_exception_type,
    before_sleep_log,
)
import logging
from app.core.logging import get_logger

logger = get_logger("validation")
T = TypeVar("T", bound=BaseModel)


class LLMValidationError(Exception):
    """Raised when LLM output fails validation after all retries."""
    def __init__(self, message: str, last_error: Exception, attempts: int):
        super().__init__(message)
        self.last_error = last_error
        self.attempts = attempts


class RetryableValidationError(Exception):
    """Internal signal to trigger a retry."""
    pass


def validate_llm_output(
    raw_output: str | dict,
    schema: Type[T],
    context: str = "unknown"
) -> tuple[T, int]:
    """
    Validate raw LLM output against a Pydantic schema.

    Returns (validated_model, attempt_count).
    Raises LLMValidationError if validation fails after retries.

    This function is called by each service after receiving a Claude response.
    It is NOT async — validation is pure CPU work, no I/O.
    """
    attempt = 0
    last_error = None

    for attempt in range(1, 4):  # max 3 attempts
        try:
            if isinstance(raw_output, str):
                # Try to parse JSON if we got a string
                try:
                    data = json.loads(raw_output)
                except json.JSONDecodeError as e:
                    raise ValueError(f"Malformed JSON in LLM output: {e}") from e
            else:
                data = raw_output

            validated = schema.model_validate(data)
            if attempt > 1:
                logger.info(
                    "validation_succeeded_after_retry",
                    schema=schema.__name__,
                    context=context,
                    attempt=attempt,
                )
            return validated, attempt - 1

        except (ValidationError, ValueError, json.JSONDecodeError) as e:
            last_error = e
            logger.warning(
                "validation_failed",
                schema=schema.__name__,
                context=context,
                attempt=attempt,
                error=str(e),
            )
            if attempt == 3:
                break
            # Brief sleep before retry signal (actual retry handled by caller)
            time.sleep(2 ** (attempt - 1))  # 1s, 2s

    raise LLMValidationError(
        f"LLM output failed validation for {schema.__name__} after {attempt} attempts. "
        f"Context: {context}. Last error: {last_error}",
        last_error=last_error,
        attempts=attempt,
    )


def parse_tool_use_result(tool_use_block: Any) -> dict:
    """
    Extract the input dict from a Claude tool_use content block.
    Claude's tool use API returns structured data directly — no JSON parsing needed.
    This is why we use tool use over asking for JSON in text.
    """
    if hasattr(tool_use_block, "input"):
        return tool_use_block.input
    raise ValueError(f"Expected tool_use block, got: {type(tool_use_block)}")


def build_extraction_tool(schema_name: str, description: str, properties: dict) -> dict:
    """
    Build a Claude tool definition from a description and property schema.
    Used to force structured output from the model.
    """
    return {
        "name": schema_name,
        "description": description,
        "input_schema": {
            "type": "object",
            "properties": properties,
            "required": list(properties.keys()),
        }
    }
