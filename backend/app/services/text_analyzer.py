"""
Text Analysis Service
---------------------
Classifies news text as REAL, FAKE, or MISLEADING using a HuggingFace
transformer model, augmented by regex-based heuristics.

Model selection (set MODEL_CHOICE env var, default = "tiny"):
  "tiny"   → mrm8488/bert-tiny-finetuned-fake-news-detection   (~17 MB)
             Fastest, lowest RAM. Best for free-tier hosting (Render 512 MB).
  "small"  → valurank/distilroberta-fake-news                  (~82 MB)
             Good accuracy, still fits free tier comfortably.
  "medium" → cross-encoder/nli-deberta-v3-xsmall               (~70 MB)
             Zero-shot model — no fine-tuning on fake-news, but generalises well.
  "large"  → facebook/bart-large-mnli                          (~1.6 GB)
             Highest accuracy. Requires 2 GB+ RAM (paid hosting plan).

Falls back to pure rule-based heuristics if any model fails to load.
"""

import os
import re
from dotenv import load_dotenv

load_dotenv(override=True)  # Always read from .env — overrides any shell-level env vars
import logging
from typing import List, Tuple

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Model registry — maps short name → HuggingFace model ID + pipeline task
# ---------------------------------------------------------------------------
MODEL_OPTIONS = {
    "tiny": {
        "model_id": "mrm8488/bert-tiny-finetuned-fake-news-detection",
        "task":     "text-classification",
        "size":     "~17 MB",
        "notes":    "Fastest. Best for free-tier (512 MB RAM). Purpose-trained on fake news.",
    },
    "small": {
        "model_id": "valurank/distilroberta-fake-news",
        "task":     "text-classification",
        "size":     "~82 MB",
        "notes":    "Better accuracy than tiny, still fits free tier.",
    },
    "medium": {
        "model_id": "cross-encoder/nli-deberta-v3-xsmall",
        "task":     "zero-shot-classification",
        "size":     "~70 MB",
        "notes":    "Zero-shot NLI model. Versatile — works on any topic.",
    },
    "large": {
        "model_id": "facebook/bart-large-mnli",
        "task":     "zero-shot-classification",
        "size":     "~1.6 GB",
        "notes":    "Highest accuracy. Needs 2 GB+ RAM. Use paid hosting.",
    },
}

# Read choice from environment — default to "tiny" for safety
_MODEL_CHOICE = os.getenv("MODEL_CHOICE", "tiny").lower()
if _MODEL_CHOICE not in MODEL_OPTIONS:
    logger.warning(
        f"Unknown MODEL_CHOICE='{_MODEL_CHOICE}'. "
        f"Valid options: {list(MODEL_OPTIONS)}. Falling back to 'tiny'."
    )
    _MODEL_CHOICE = "tiny"

_ACTIVE_MODEL = MODEL_OPTIONS[_MODEL_CHOICE]
logger.info(
    f"Model selected: '{_MODEL_CHOICE}' → {_ACTIVE_MODEL['model_id']} "
    f"({_ACTIVE_MODEL['size']})"
)

# Singleton pipeline — loaded once on first use
_pipeline = None


def _get_pipeline():
    """Lazy-load the transformer pipeline. Returns None if load fails."""
    global _pipeline
    if _pipeline is None:
        try:
            from transformers import pipeline
            _pipeline = pipeline(
                _ACTIVE_MODEL["task"],
                model=_ACTIVE_MODEL["model_id"],
            )
            logger.info(
                f"Pipeline loaded: {_ACTIVE_MODEL['model_id']} "
                f"({_ACTIVE_MODEL['size']})"
            )
        except Exception as exc:
            logger.warning(
                f"Could not load model '{_ACTIVE_MODEL['model_id']}': {exc}. "
                "Falling back to rule-based heuristics."
            )
    return _pipeline


def get_active_model_info() -> dict:
    """Return info about the currently selected model (for /health endpoint)."""
    return {
        "choice":   _MODEL_CHOICE,
        "model_id": _ACTIVE_MODEL["model_id"],
        "task":     _ACTIVE_MODEL["task"],
        "size":     _ACTIVE_MODEL["size"],
        "loaded":   _pipeline is not None,
    }


# ---------------------------------------------------------------------------
# Misinformation signal patterns (rule-based augmentation / fallback)
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
    seen, unique = set(), []
    for p in phrases:
        if p not in seen:
            seen.add(p)
            unique.append(p)
    return unique[:5]


# ---------------------------------------------------------------------------
# Heuristic fallback (no model needed)
# ---------------------------------------------------------------------------

def _heuristic_analysis(text: str) -> Tuple[str, float, int, str]:
    """
    Rule-based analysis when the transformer model is unavailable.
    Returns (label, confidence, credibility_score, explanation).
    """
    fake_hits      = _count_pattern_hits(text, FAKE_PATTERNS)
    credible_hits  = _count_pattern_hits(text, CREDIBLE_SIGNALS)
    misleading_hits = _count_pattern_hits(text, MISLEADING_PATTERNS)

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
# Model inference helpers — one per task type
# ---------------------------------------------------------------------------

def _run_text_classification(pipe, text: str) -> Tuple[str, float, int, str]:
    """
    Handle output from text-classification pipelines.
    Used by: 'tiny' (bert-tiny) and 'small' (distilroberta) models.
    Label convention: LABEL_0 = FAKE, LABEL_1 = REAL
    """
    result     = pipe(text[:512], truncation=True)[0]
    raw_label  = result["label"]   # "LABEL_0" or "LABEL_1"
    score      = round(result["score"], 3)

    # Map to our labels — low-confidence predictions → MISLEADING
    if raw_label == "LABEL_0":          # model says REAL
        label = "REAL" if score >= 0.65 else "MISLEADING"
    else:                               # model says FAKE
        label = "FAKE" if score >= 0.65 else "MISLEADING"

    confidence = score
    credibility_score = _score_from_label(label, confidence)
    explanation = (
        f"AI model ({_ACTIVE_MODEL['model_id']}) classified this as "
        f"'{label}' with {confidence * 100:.1f}% confidence. "
        + _build_explanation(label, confidence)
    )
    return label, confidence, credibility_score, explanation


def _run_zero_shot_classification(pipe, text: str) -> Tuple[str, float, int, str]:
    """
    Handle output from zero-shot-classification pipelines.
    Used by: 'medium' (deberta-xsmall) and 'large' (bart-large-mnli) models.
    """
    candidate_labels = ["real news", "fake news", "misleading content"]
    result   = pipe(text[:1024], candidate_labels)

    label_map = {
        "real news":         "REAL",
        "fake news":         "FAKE",
        "misleading content": "MISLEADING",
    }
    top_label  = result["labels"][0]
    confidence = round(result["scores"][0], 3)
    label      = label_map.get(top_label, "UNKNOWN")

    credibility_score = _score_from_label(label, confidence)
    explanation = (
        f"AI model ({_ACTIVE_MODEL['model_id']}) classified this as "
        f"'{label}' with {confidence * 100:.1f}% confidence. "
        + _build_explanation(label, confidence)
    )
    return label, confidence, credibility_score, explanation


def _score_from_label(label: str, confidence: float) -> int:
    """Convert label + confidence into a 0–100 credibility score."""
    if label == "REAL":
        score = int(50 + confidence * 50)
    elif label == "MISLEADING":
        score = int(50 - confidence * 30)
    else:  # FAKE
        score = int(50 - confidence * 45)
    return max(0, min(100, score))


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
            "label":            "UNKNOWN",
            "confidence":       0.0,
            "credibility_score": 0,
            "explanation":      "Input text is too short to analyse.",
            "key_phrases":      [],
            "sources":          [],
        }

    pipe = _get_pipeline()

    if pipe:
        try:
            task = _ACTIVE_MODEL["task"]
            if task == "text-classification":
                label, confidence, credibility_score, explanation = \
                    _run_text_classification(pipe, text)
            else:  # zero-shot-classification
                label, confidence, credibility_score, explanation = \
                    _run_zero_shot_classification(pipe, text)

        except Exception as exc:
            logger.error(f"Pipeline inference failed: {exc}. Using heuristics.")
            label, confidence, credibility_score, explanation = \
                _heuristic_analysis(text)
    else:
        label, confidence, credibility_score, explanation = \
            _heuristic_analysis(text)

    # Augment credibility score with rule-based signals
    fake_hits     = _count_pattern_hits(text, FAKE_PATTERNS)
    credible_hits = _count_pattern_hits(text, CREDIBLE_SIGNALS)
    credibility_score = max(
        0, min(100, credibility_score - fake_hits * 3 + credible_hits * 2)
    )

    return {
        "label":             label,
        "confidence":        confidence,
        "credibility_score": credibility_score,
        "explanation":       explanation,
        "key_phrases":       _extract_key_phrases(text, label),
        "sources":           _build_sources(label),
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
            "https://www.boomlive.in",    # India-focused
            "https://www.altnews.in",     # India-focused
        ]
    return base
