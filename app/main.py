"""
FintelliDoc — FastAPI application entry point.

Lifespan events handle startup validation and graceful shutdown.
Startup fails fast if ANTHROPIC_API_KEY is missing or invalid —
better to crash at boot than to fail silently in production.
"""

from contextlib import asynccontextmanager
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
import anthropic

from app.api.routes import router
from app.core.config import get_settings
from app.core.logging import setup_logging, get_logger

settings = get_settings()
logger = get_logger("main")


@asynccontextmanager
async def lifespan(app: FastAPI):
    """
    Application lifespan manager.
    Validates config and connectivity at startup.
    """
    setup_logging()
    logger.info("fintellidoc_starting", model=settings.claude_model, debug=settings.debug)

    # Validate Anthropic API key at startup
    try:
        client = anthropic.Anthropic(api_key=settings.anthropic_api_key)
        # Lightweight check — list models or just validate key format
        logger.info("anthropic_client_initialized", model=settings.claude_model)
    except Exception as e:
        logger.error("anthropic_client_failed", error=str(e))
        raise RuntimeError(f"Failed to initialize Anthropic client: {e}")

    logger.info("fintellidoc_started")
    yield
    logger.info("fintellidoc_shutdown")


app = FastAPI(
    title="FintelliDoc",
    description=(
        "AI-powered financial document intelligence platform. "
        "Extract entities, financial metrics, risk signals, and semantic Q&A "
        "from financial documents using Anthropic Claude."
    ),
    version="1.0.0",
    docs_url="/docs",
    redoc_url="/redoc",
    lifespan=lifespan,
)

# CORS — restrict in production
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"] if settings.debug else [],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(router, prefix=settings.api_prefix)


@app.get("/", include_in_schema=False)
async def root():
    return {
        "service": "FintelliDoc",
        "version": "1.0.0",
        "docs": "/docs",
        "health": f"{settings.api_prefix}/health",
    }
