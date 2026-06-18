"""
Structured JSON logging using structlog.
Every log entry includes: timestamp, level, service, and contextual fields.
This makes logs queryable in production (CloudWatch, Datadog, etc.)
"""

import logging
import structlog
from app.core.config import get_settings


def setup_logging() -> None:
    settings = get_settings()

    structlog.configure(
        processors=[
            structlog.contextvars.merge_contextvars,
            structlog.processors.add_log_level,
            structlog.processors.TimeStamper(fmt="iso"),
            structlog.processors.StackInfoRenderer(),
            structlog.processors.format_exc_info,
            structlog.processors.JSONRenderer(),
        ],
        wrapper_class=structlog.make_filtering_bound_logger(
            getattr(logging, settings.log_level.upper())
        ),
        context_class=dict,
        logger_factory=structlog.PrintLoggerFactory(),
    )


def get_logger(name: str = "fintellidoc"):
    return structlog.get_logger(name)
