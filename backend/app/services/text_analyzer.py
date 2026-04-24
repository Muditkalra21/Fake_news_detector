"""
Text Analysis Service
---------------------
Uses a HuggingFace zero-shot classification pipeline to classify
news text as REAL, FAKE, or MISLEADING.  Falls back to a heuristic
rule-based approach when the model is unavailable (e.g. no internet).
"""

import re
import logging
from typing import List, Tuple

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Attempt to load the transformer pipeline once at import time.
# We use a lightweight model so startup stays fast.
# ---------------------------------------------------------------------------
_pipeline = None

def _get_pipeline():
    global _pipeline
    if _pipeline is None:
        try:
            from transformers import pipeline
            # mrm8488/bert-tiny-finetuned-fake-news-detection ≈ 17 MB
            # Purpose-trained REAL/FAKE classifier — 4× smaller than the
            # zero-shot model, loads in ~2s, fits comfortably in 512 MB RAM.
            # Upgrade option (better accuracy, still small):
            #   "valurank/distilroberta-fake-news"  ≈ 82 MB
            _pipeline = pipeline(
                "text-classification",
                model="mrm8488/bert-tiny-finetuned-fake-news-detection",
            )
            logger.info("Transformer pipeline loaded successfully (bert-tiny, ~17 MB).")
        except Exception as exc:
            logger.warning(f"Could not load transformer pipeline: {exc}")
    return _pipeline


# ---------------------------------------------------------------------------
# Misinformation signal patterns (rule-based fallback / augmentation)
# ---------------------------------------------------------------------------
FAKE_PATTERNS: List[str] = [
    r"\bbreaking[:\s]+",
    r"\bshock(ing)?\b",
    r"\byou won'?t believe\b",
    r"\bexclusive[:\s]+",
    r"\b(scientists|doctors|experts) (don'?t|hate|fear)\b",
    r"\bconspiracy\b",
    r"\bdeep.?state\b",
    r"\bplandemic\b",
    r"\bsecret (cure|remedy|vaccine)\b",
    r"\b100%\s*(proven|guaranteed|effective)\b",
    r"\bclickbait\b",
    r"\bGOVERNMENT IS HIDING\b",
    r"\bthey don'?t want you to know\b",
]

CREDIBLE_SIGNALS: List[str] = [
    r"\baccording to\b",
    r"\bstudy (shows|finds|published)\b",
    r"\bpeer-reviewed\b",
    r"\bsources (say|confirm|report)\b",
    r"\bofficial statement\b",
    r"\bpress release\b",
    r"\bgovernment (confirmed|announced)\b",
]

MISLEADING_PATTERNS: List[str] = [
    r"\bout of context\b",
    r"\bmisleading\b",
    r"\bpartially (true|false)\b",
    r"\bsatire\b",
    r"\bparody\b",
]


def _count_pattern_hits(text: str, patterns: List[str]) -> int:
    """Count how many regex patterns match in the text."""
    text_lower = text.lower()
    return sum(1 for p in patterns if re.search(p, text_lower))


def _extract_key_phrases(text: str, label: str) -> List[str]:
    """Pull out suspicious or notable phrases for the explainability panel."""
    phrases: List[str] = []
    text_lower = text.lower()

    if label in ("FAKE", "MISLEADING"):
        for pattern in FAKE_PATTERNS + MISLEADING_PATTERNS:
            match = re.search(pattern, text_lower)
            if match:
                start = max(0, match.start() - 10)
                end = min(len(text), match.end() + 10)
                phrases.append(f'…{text[start:end].strip()}…')
    else:
        for pattern in CREDIBLE_SIGNALS:
            match = re.search(pattern, text_lower)
            if match:
                start = max(0, match.start() - 5)
                end = min(len(text), match.end() + 20)
                phrases.append(f'…{text[start:end].strip()}…')

    # Deduplicate while preserving order
    seen = set()
    unique = []
    for p in phrases:
        if p not in seen:
            seen.add(p)
            unique.append(p)
    return unique[:5]


def _heuristic_analysis(text: str) -> Tuple[str, float, int, str]:
    """
    Rule-based fallback when the transformer model is unavailable.
    Returns (label, confidence, credibility_score, explanation).
    """
    fake_hits = _count_pattern_hits(text, FAKE_PATTERNS)
    credible_hits = _count_pattern_hits(text, CREDIBLE_SIGNALS)
    misleading_hits = _count_pattern_hits(text, MISLEADING_PATTERNS)

    total = fake_hits + credible_hits + misleading_hits or 1

    if misleading_hits > 0 and misleading_hits >= fake_hits:
        label = "MISLEADING"
        confidence = min(0.55 + misleading_hits * 0.08, 0.90)
        credibility_score = max(10, 50 - fake_hits * 8 + credible_hits * 5)
        explanation = (
            f"Content contains {misleading_hits} misleading signal(s) "
            f"and {fake_hits} sensationalist pattern(s). "
            "It may present facts selectively or out of context."
        )
    elif fake_hits > credible_hits:
        label = "FAKE"
        confidence = min(0.50 + fake_hits * 0.10, 0.95)
        credibility_score = max(5, 40 - fake_hits * 10 + credible_hits * 5)
        explanation = (
            f"Detected {fake_hits} sensationalist / misinformation pattern(s). "
            "Language typical of misleading or fabricated content."
        )
    else:
        label = "REAL"
        confidence = min(0.50 + credible_hits * 0.10, 0.92)
        credibility_score = min(95, 55 + credible_hits * 8 - fake_hits * 5)
        explanation = (
            f"Content shows {credible_hits} credibility signal(s) "
            "and limited sensationalist language."
        )

    return label, confidence, credibility_score, explanation


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def analyze_text(text: str) -> dict:
    """
    Analyse a block of news text and return a structured result dict.

    Returns keys: label, confidence, credibility_score, explanation,
                  key_phrases, sources
    """
    if not text or len(text.strip()) < 20:
        return {
            "label": "UNKNOWN",
            "confidence": 0.0,
            "credibility_score": 0,
            "explanation": "Input text is too short to analyse.",
            "key_phrases": [],
            "sources": [],
        }

    pipe = _get_pipeline()

    if pipe:
        try:
            result = pipe(text[:512], truncation=True)[0]  # returns {label, score}

            # bert-tiny: LABEL_1 = REAL, LABEL_0 = FAKE
            raw_label = result["label"]   # "LABEL_0" or "LABEL_1"
            score = round(result["score"], 3)

            if raw_label == "LABEL_1":          # model says REAL
                if score >= 0.65:
                    label = "REAL"
                else:
                    label = "MISLEADING"        # uncertain → cautious middle ground
            else:                               # model says FAKE
                if score >= 0.65:
                    label = "FAKE"
                else:
                    label = "MISLEADING"

            confidence = score

            # Compute credibility score (0-100)
            if label == "REAL":
                credibility_score = int(50 + confidence * 50)
            elif label == "MISLEADING":
                credibility_score = int(50 - confidence * 30)
            else:  # FAKE
                credibility_score = int(50 - confidence * 45)

            credibility_score = max(0, min(100, credibility_score))

            explanation = (
                f"AI model classified this as '{label}' with "
                f"{confidence * 100:.1f}% confidence. "
                + _build_explanation(label, confidence)
            )

        except Exception as exc:
            logger.error(f"Pipeline inference failed: {exc}. Using heuristics.")
            label, confidence, credibility_score, explanation = _heuristic_analysis(text)
    else:
        label, confidence, credibility_score, explanation = _heuristic_analysis(text)

    # Augment credibility score with rule-based signals
    fake_hits = _count_pattern_hits(text, FAKE_PATTERNS)
    credible_hits = _count_pattern_hits(text, CREDIBLE_SIGNALS)
    credibility_score = max(0, min(100, credibility_score - fake_hits * 3 + credible_hits * 2))

    key_phrases = _extract_key_phrases(text, label)
    sources = _build_sources(label)

    return {
        "label": label,
        "confidence": confidence,
        "credibility_score": credibility_score,
        "explanation": explanation,
        "key_phrases": key_phrases,
        "sources": sources,
    }


def _build_explanation(label: str, confidence: float) -> str:
    """Generate a human-readable explanation suffix."""
    if label == "FAKE":
        return (
            "The content exhibits patterns commonly found in fabricated stories, "
            "including sensationalist language, unverified claims, and emotional triggers."
        )
    elif label == "MISLEADING":
        return (
            "While parts of the content may be factual, key context appears missing "
            "or the framing may lead readers to incorrect conclusions."
        )
    else:
        return (
            "The content uses measured language and appears to cite verifiable sources. "
            "Always cross-check with additional trusted outlets."
        )


def _build_sources(label: str) -> List[str]:
    """Return relevant fact-checking resources based on verdict."""
    base = [
        "https://www.snopes.com",
        "https://www.factcheck.org",
        "https://www.politifact.com",
    ]
    if label in ("FAKE", "MISLEADING"):
        return base + [
            "https://toolbox.google.com/factcheck/explorer",
            "https://www.boomlive.in",          # India-focused
            "https://www.altnews.in",            # India-focused
        ]
    return base
