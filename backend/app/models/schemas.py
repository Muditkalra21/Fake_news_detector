"""
Pydantic schemas for request/response validation.
"""
from pydantic import BaseModel
from typing import Optional, List, Any


class TextAnalysisRequest(BaseModel):
    """Request body for text-based fake news analysis."""
    text: str
    language: Optional[str] = "auto"  # "auto" = detect automatically


class FactCheckResult(BaseModel):
    """A single journalist fact-check from Google Fact Check API."""
    claim:     str
    claimant:  str
    verdict:   str
    publisher: str
    url:       str


class NewsArticle(BaseModel):
    """A single news article from NewsAPI.org."""
    title:     str
    source:    str
    url:       str
    published: str
    credible:  bool


class AnalysisResult(BaseModel):
    """Unified result returned for any analysis type."""
    label:             str            # "REAL", "FAKE", or "MISLEADING"
    confidence:        float          # 0.0 – 1.0
    credibility_score: int            # 0 – 100
    explanation:       str            # Human-readable reason
    key_phrases:       List[str]      # Flagged phrases / observations
    sources:           List[str]      # Fact-check references / supporting links
    detected_language: Optional[str] = None

    # External API enrichment fields
    fact_checks:       List[Any] = []   # Journalist fact-checks (Google)
    news_articles:     List[Any] = []   # News coverage (NewsAPI)
    external_checked:  bool = False     # Whether external APIs were queried
    score_adjusted:    bool = False     # Whether score was adjusted by APIs


class VideoAnalysisRequest(BaseModel):
    """Request body for video link analysis."""
    url: str
    description: Optional[str] = ""
