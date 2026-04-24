"""
Video Analysis Service
----------------------
Analyses a video URL for credibility signals:
  - Source / domain reputation
  - URL structure red flags
  - Platform detection (YouTube, Vimeo, etc.)
  - Description text analysis (re-uses text_analyzer)
  - Known fact-check databases (heuristic)

For deepfake detection on actual video frames, integrate a model like
XceptionNet or EfficientNet trained on FaceForensics++.
"""

import re
import logging
from urllib.parse import urlparse
from typing import List

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Domain reputation lists
# ---------------------------------------------------------------------------

TRUSTED_DOMAINS = {
    # International news
    "bbc.com", "bbc.co.uk", "reuters.com", "apnews.com",
    "theguardian.com", "nytimes.com", "washingtonpost.com",
    "economist.com", "ft.com", "bloomberg.com",
    # Indian mainstream
    "thehindu.com", "hindustantimes.com", "ndtv.com",
    "timesofindia.com", "indianexpress.com", "thewire.in",
    "scroll.in", "theprint.in", "livemint.com",
    # Fact-checkers
    "snopes.com", "factcheck.org", "politifact.com",
    "altnews.in", "boomlive.in", "vishvasnews.com",
    # Video platforms
    "youtube.com", "youtu.be", "vimeo.com",
}

SUSPICIOUS_DOMAINS = {
    "beforeitsnews.com", "naturalnews.com", "infowars.com",
    "worldnewsdailyreport.com", "empirenews.net",
    "huzlers.com", "thelastlineofdefense.org",
    "newslo.com", "dailybuzzlive.com",
}

# URL patterns that are often associated with low-quality / spam content
SUSPICIOUS_URL_PATTERNS: List[str] = [
    r"\d{4}/\d{2}/\d{2}/.{3,30}-you-won.?t-believe",
    r"(shocking|explosive|bombshell|viral|exclusive)[-_]",
    r"[-_](hoax|fake|scam|fraud|conspiracy)",
    r"\?utm_source=facebook",              # often viral bait
]


# ---------------------------------------------------------------------------
# YouTube / video platform helpers
# ---------------------------------------------------------------------------

def _detect_platform(url: str) -> str:
    """Identify the hosting platform from the URL."""
    if "youtube.com" in url or "youtu.be" in url:
        return "YouTube"
    if "vimeo.com" in url:
        return "Vimeo"
    if "dailymotion.com" in url:
        return "Dailymotion"
    if "twitter.com" in url or "x.com" in url:
        return "X / Twitter"
    if "facebook.com" in url or "fb.watch" in url:
        return "Facebook"
    if "instagram.com" in url:
        return "Instagram"
    if "tiktok.com" in url:
        return "TikTok"
    return "Unknown"


def _extract_domain(url: str) -> str:
    try:
        parsed = urlparse(url)
        domain = parsed.netloc.lower()
        # Strip www. prefix
        return domain.removeprefix("www.")
    except Exception:
        return ""


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def analyze_video(url: str, description: str = "") -> dict:
    """
    Analyse a video URL and optional description for credibility.

    Returns: label, confidence, credibility_score, explanation,
             key_phrases, sources
    """
    observations: List[str] = []
    trust_score = 50  # start neutral (0-100 scale)

    domain = _extract_domain(url)
    platform = _detect_platform(url)

    observations.append(f"Platform detected: {platform}")

    # --- Domain reputation ---
    if domain in TRUSTED_DOMAINS:
        trust_score += 25
        observations.append(f"Source domain '{domain}' is in the trusted list")
    elif domain in SUSPICIOUS_DOMAINS:
        trust_score -= 30
        observations.append(f"⚠️ Source domain '{domain}' is flagged as unreliable")
    elif domain:
        observations.append(f"Domain '{domain}' has no known reputation record")

    # --- URL pattern check ---
    url_lower = url.lower()
    suspicious_url_hits = 0
    for pattern in SUSPICIOUS_URL_PATTERNS:
        if re.search(pattern, url_lower):
            suspicious_url_hits += 1
            observations.append(f"Suspicious URL pattern matched: '{pattern}'")
    trust_score -= suspicious_url_hits * 10

    # --- Description text analysis ---
    if description and len(description.strip()) > 20:
        from app.services.text_analyzer import analyze_text, _count_pattern_hits, FAKE_PATTERNS, CREDIBLE_SIGNALS
        fake_hits = _count_pattern_hits(description, FAKE_PATTERNS)
        credible_hits = _count_pattern_hits(description, CREDIBLE_SIGNALS)
        trust_score += credible_hits * 5 - fake_hits * 8
        if fake_hits:
            observations.append(f"Description contains {fake_hits} sensationalist phrase(s)")
        if credible_hits:
            observations.append(f"Description contains {credible_hits} credibility signal(s)")

    # --- Clamp score ---
    trust_score = max(0, min(100, trust_score))

    # --- Verdict ---
    if trust_score >= 65:
        label = "REAL"
        confidence = round(min(0.50 + (trust_score - 65) / 100, 0.92), 3)
        explanation = (
            "The video source and URL appear credible. "
            "Content originates from a reputable platform or outlet."
        )
    elif trust_score >= 38:
        label = "MISLEADING"
        confidence = round(min(0.45 + (50 - trust_score) / 100, 0.85), 3)
        explanation = (
            "The video source or URL shows some credibility concerns. "
            "The content may be real but presented out of context."
        )
    else:
        label = "FAKE"
        confidence = round(min(0.50 + (38 - trust_score) / 80, 0.95), 3)
        explanation = (
            "Multiple red flags detected: suspicious domain, URL patterns, "
            "or description language typical of misinformation."
        )

    return {
        "label": label,
        "confidence": confidence,
        "credibility_score": trust_score,
        "explanation": explanation,
        "key_phrases": observations[:6],
        "sources": [
            "https://www.invid-project.eu",
            "https://www.tineye.com",
            "https://toolbox.google.com/factcheck/explorer",
            "https://www.snopes.com",
            "https://boomlive.in",
        ],
    }
