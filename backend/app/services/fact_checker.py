"""
Fact Checker Service
---------------------
Queries external APIs to cross-verify news claims:

  1. Google Fact Check Tools API
     → Searches a database of professional journalist fact-checks
     → Returns matching claims with verdicts (True / False / Misleading)

  2. NewsAPI.org
     → Searches 150,000+ news outlets for coverage of the claim
     → Real news stories are usually covered by multiple credible outlets

Both services are OPTIONAL — if the API keys are missing or a request
fails, the function returns empty results gracefully (no crash).

Score impact (applied in analysis.py on top of LLM score):
  Google: each "false/fake" verdict  → -12 pts
          each "true/confirmed"       → +8  pts
  NewsAPI: 5+ credible articles       → +10 pts
           0 credible articles        → -8  pts
           1-4 articles               → neutral
"""

import os
import re
import logging
import httpx
from typing import List, Optional
from dotenv import load_dotenv

load_dotenv(override=True)

logger = logging.getLogger(__name__)

GOOGLE_FACTCHECK_API_KEY = os.getenv("GOOGLE_FACTCHECK_API_KEY", "")
NEWS_API_KEY = os.getenv("NEWS_API_KEY", "")

GOOGLE_FACTCHECK_URL = "https://factchecktools.googleapis.com/v1alpha1/claims:search"
NEWS_API_URL = "https://newsapi.org/v2/everything"

# Verdicts that indicate the claim is false/fake
FALSE_VERDICTS = {
    "false", "mostly false", "fake", "incorrect", "misleading",
    "pants on fire", "inaccurate", "fabricated", "debunked",
    "no evidence", "unverified", "disputed"
}
# Verdicts that indicate the claim is true
TRUE_VERDICTS = {
    "true", "mostly true", "correct", "accurate", "confirmed",
    "verified", "supported", "factual"
}

REQUEST_TIMEOUT = 8.0  # seconds


# ---------------------------------------------------------------------------
# Data structures (plain dicts — no heavy Pydantic here)
# ---------------------------------------------------------------------------

def _empty_result() -> dict:
    return {
        "fact_checks": [],       # List of journalist fact-checks
        "news_articles": [],     # List of news articles covering the claim
        "score_delta": 0,        # How much to adjust credibility score (+/-)
        "has_data": False,       # Whether any external data was found
    }


# ---------------------------------------------------------------------------
# Google Fact Check Tools API
# ---------------------------------------------------------------------------

def _query_google_factcheck(query: str) -> List[dict]:
    """
    Query the Google Fact Check Tools API.
    Returns a list of fact-check results, each containing:
      - claim: the original claim text
      - claimant: who made the claim
      - verdict: rating text (e.g. "False")
      - publisher: name of the fact-checking organization
      - url: link to the full fact-check article
    """
    if not GOOGLE_FACTCHECK_API_KEY or GOOGLE_FACTCHECK_API_KEY == "your_api_key_here":
        logger.warning("[FactCheck] GOOGLE_FACTCHECK_API_KEY not set — skipping Google check")
        return []

    # Use first 200 chars as the search query (API limit)
    search_query = query[:200].strip()

    try:
        logger.info(f"[FactCheck] Querying Google Fact Check API | query={search_query[:60]}...")
        with httpx.Client(timeout=REQUEST_TIMEOUT) as client:
            response = client.get(
                GOOGLE_FACTCHECK_URL,
                params={
                    "key": GOOGLE_FACTCHECK_API_KEY,
                    "query": search_query,
                    "languageCode": "en",
                    "pageSize": 5,
                },
            )
            response.raise_for_status()
            data = response.json()

        claims = data.get("claims", [])
        results = []

        for claim in claims:
            reviews = claim.get("claimReview", [])
            for review in reviews:
                verdict_text = review.get("textualRating", "Unknown").strip()
                results.append({
                    "claim":     claim.get("text", "")[:300],
                    "claimant":  claim.get("claimant", "Unknown"),
                    "verdict":   verdict_text,
                    "publisher": review.get("publisher", {}).get("name", "Unknown"),
                    "url":       review.get("url", ""),
                })

        logger.info(f"[FactCheck] Google returned {len(results)} fact-check(s)")
        return results

    except httpx.HTTPStatusError as e:
        logger.error(f"[FactCheck] Google API HTTP error: {e.response.status_code} — {e.response.text[:200]}")
        return []
    except Exception as e:
        logger.error(f"[FactCheck] Google API request failed: {e}")
        return []


# ---------------------------------------------------------------------------
# NewsAPI.org
# ---------------------------------------------------------------------------

def _extract_keywords(text: str, max_words: int = 6) -> str:
    """Extract the most meaningful keywords from the claim for NewsAPI search."""
    # Remove common stop words and punctuation
    stop_words = {
        "the", "a", "an", "is", "are", "was", "were", "be", "been",
        "being", "have", "has", "had", "do", "does", "did", "will",
        "would", "could", "should", "may", "might", "shall", "can",
        "in", "on", "at", "to", "for", "of", "and", "or", "but",
        "with", "this", "that", "these", "those", "it", "its", "not",
        "no", "new", "say", "says", "said", "claim", "claims",
    }
    words = re.findall(r'\b[a-zA-Z]{4,}\b', text)
    keywords = [w for w in words if w.lower() not in stop_words][:max_words]
    return " ".join(keywords)


# Well-known credible news domains
CREDIBLE_DOMAINS = {
    "bbc.com", "bbc.co.uk", "reuters.com", "apnews.com", "theguardian.com",
    "nytimes.com", "washingtonpost.com", "cnn.com", "nbcnews.com",
    "abcnews.go.com", "cbsnews.com", "npr.org", "theatlantic.com",
    "bloomberg.com", "ft.com", "economist.com", "time.com", "forbes.com",
    "ndtv.com", "thehindu.com", "hindustantimes.com", "indianexpress.com",
    "aljazeera.com", "dw.com", "france24.com",
}


def _is_credible_source(url: str) -> bool:
    """Check whether a URL is from a credible news outlet."""
    try:
        domain = url.lower().split("//")[-1].split("/")[0].replace("www.", "")
        return any(domain == d or domain.endswith("." + d) for d in CREDIBLE_DOMAINS)
    except Exception:
        return False


def _query_newsapi(text: str) -> List[dict]:
    """
    Query NewsAPI.org for news articles related to the claim.
    Returns a list of matching articles with source, title, and URL.
    """
    if not NEWS_API_KEY or NEWS_API_KEY == "your_news_api_key_here":
        logger.warning("[NewsAPI] NEWS_API_KEY not set — skipping NewsAPI check")
        return []

    keywords = _extract_keywords(text)
    if not keywords:
        return []

    try:
        logger.info(f"[NewsAPI] Querying NewsAPI | keywords={keywords}")
        with httpx.Client(timeout=REQUEST_TIMEOUT) as client:
            response = client.get(
                NEWS_API_URL,
                params={
                    "apiKey":   NEWS_API_KEY,
                    "q":        keywords,
                    "language": "en",
                    "sortBy":   "relevancy",
                    "pageSize": 10,
                },
            )
            response.raise_for_status()
            data = response.json()

        articles = data.get("articles", [])
        results = []

        for article in articles:
            url = article.get("url", "")
            source_name = article.get("source", {}).get("name", "Unknown")
            results.append({
                "title":      (article.get("title") or "")[:200],
                "source":     source_name,
                "url":        url,
                "published":  article.get("publishedAt", "")[:10],
                "credible":   _is_credible_source(url),
            })

        logger.info(
            f"[NewsAPI] Returned {len(results)} article(s), "
            f"{sum(1 for a in results if a['credible'])} from credible sources"
        )
        return results

    except httpx.HTTPStatusError as e:
        logger.error(f"[NewsAPI] HTTP error: {e.response.status_code} — {e.response.text[:200]}")
        return []
    except Exception as e:
        logger.error(f"[NewsAPI] Request failed: {e}")
        return []


# ---------------------------------------------------------------------------
# Score delta calculation
# ---------------------------------------------------------------------------

def _calculate_score_delta(fact_checks: List[dict], news_articles: List[dict]) -> int:
    """
    Calculate how much to adjust the LLM's credibility score (+/-).
    Based on fact-check verdicts and news coverage quality.
    """
    delta = 0

    # ── Google Fact Check scoring ──────────────────────────────────────────
    for fc in fact_checks:
        verdict_lower = fc["verdict"].lower()
        if any(v in verdict_lower for v in FALSE_VERDICTS):
            delta -= 12
            logger.info(f"[ScoreDelta] -12 pts: fact-check verdict '{fc['verdict']}' by {fc['publisher']}")
        elif any(v in verdict_lower for v in TRUE_VERDICTS):
            delta += 8
            logger.info(f"[ScoreDelta] +8 pts: fact-check verdict '{fc['verdict']}' by {fc['publisher']}")

    # ── NewsAPI scoring ────────────────────────────────────────────────────
    credible_count = sum(1 for a in news_articles if a["credible"])

    if credible_count >= 5:
        delta += 10
        logger.info(f"[ScoreDelta] +10 pts: {credible_count} credible news sources cover this story")
    elif credible_count == 0 and len(news_articles) > 0:
        delta -= 8
        logger.info(f"[ScoreDelta] -8 pts: no credible news sources found ({len(news_articles)} fringe sources)")
    elif credible_count == 0 and len(news_articles) == 0:
        delta -= 5
        logger.info(f"[ScoreDelta] -5 pts: zero news coverage found")
    # 1-4 credible sources → neutral (no change)

    return delta


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def run_external_checks(text: str) -> dict:
    """
    Run both Google Fact Check and NewsAPI checks on the given text.

    Returns:
        {
          "fact_checks":    [...],   # journalist fact-check results
          "news_articles":  [...],   # matching news articles
          "score_delta":    int,     # how much to adjust the LLM score
          "has_data":       bool,    # True if any external results were found
        }
    """
    result = _empty_result()

    fact_checks   = _query_google_factcheck(text)
    news_articles = _query_newsapi(text)

    result["fact_checks"]   = fact_checks
    result["news_articles"] = news_articles
    result["score_delta"]   = _calculate_score_delta(fact_checks, news_articles)
    result["has_data"]      = bool(fact_checks or news_articles)

    logger.info(
        f"[ExternalCheck] Done | fact_checks={len(fact_checks)} | "
        f"news_articles={len(news_articles)} | score_delta={result['score_delta']:+d}"
    )

    return result
