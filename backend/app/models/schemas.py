"""
Pydantic schemas for request/response validation.
"""
from pydantic import BaseModel, HttpUrl
from typing import Optional, List


class TextAnalysisRequest(BaseModel):
    """Request body for text-based fake news analysis."""
    text: str
    language: Optional[str] = "auto"  # "auto" = detect automatically


class AnalysisResult(BaseModel):
    """Unified result returned for any analysis type."""
    label: str                      # "REAL", "FAKE", or "MISLEADING"
    confidence: float               # 0.0 – 1.0
    credibility_score: int          # 0 – 100
    explanation: str                # Human-readable reason
    key_phrases: List[str]          # Flagged phrases / observations
    sources: List[str]              # Fact-check references / supporting links
    detected_language: Optional[str] = None


class VideoAnalysisRequest(BaseModel):
    """Request body for video link analysis."""
    url: str
    description: Optional[str] = ""
