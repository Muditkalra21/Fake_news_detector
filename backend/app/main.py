"""
FastAPI Application Entry Point
--------------------------------
Registers all routers, configures CORS for the React frontend,
and adds a health-check endpoint.

Environment variables:
  FRONTEND_URL  — your Vercel URL e.g. https://truthlens.vercel.app
                  (defaults to localhost for local development)
"""

import os
import logging
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from app.routers.analysis import router as analysis_router

# ---------------------------------------------------------------------------
# Logging configuration
# ---------------------------------------------------------------------------
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)-8s | %(name)s | %(message)s",
)
logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# App initialisation
# ---------------------------------------------------------------------------
app = FastAPI(
    title="Fake News Detection API",
    description=(
        "AI-powered multi-modal fake news detection system. "
        "Supports text, image, and video analysis with credibility scoring."
    ),
    version="1.0.0",
    docs_url="/docs",
    redoc_url="/redoc",
)

# ---------------------------------------------------------------------------
# CORS — allow the React dev server (and any origin in dev mode)
# ---------------------------------------------------------------------------
# Build allowed origins list — always include localhost for dev,
# plus any FRONTEND_URL set via environment variable (e.g. Vercel URL).
_allowed_origins = [
    "http://localhost:5173",
    "http://localhost:3000",
    "http://127.0.0.1:5173",
]
_frontend_url = os.getenv("FRONTEND_URL")
if _frontend_url:
    _allowed_origins.append(_frontend_url)

app.add_middleware(
    CORSMiddleware,
    allow_origins=_allowed_origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ---------------------------------------------------------------------------
# Register routers
# ---------------------------------------------------------------------------
app.include_router(analysis_router)


# ---------------------------------------------------------------------------
# Health check
# ---------------------------------------------------------------------------
@app.get("/health", tags=["system"])
async def health():
    """Quick endpoint to verify the API is running."""
    return {"status": "ok", "version": "1.0.0"}


@app.get("/", tags=["system"])
async def root():
    """API root with links to documentation."""
    return {
        "message": "Fake News Detection API",
        "docs": "/docs",
        "health": "/health",
    }
