"""
Language Detection Service
--------------------------
Detects the language of input text and supports multilingual content.
Wraps the langdetect library with a clean interface.
"""

import logging
from typing import Optional

logger = logging.getLogger(__name__)

# Language code → human-readable name (includes Indian regional languages)
LANGUAGE_NAMES = {
    "en": "English",
    "hi": "Hindi (हिन्दी)",
    "bn": "Bengali (বাংলা)",
    "te": "Telugu (తెలుగు)",
    "mr": "Marathi (मराठी)",
    "ta": "Tamil (தமிழ்)",
    "ur": "Urdu (اردو)",
    "gu": "Gujarati (ગુજરાતી)",
    "kn": "Kannada (ಕನ್ನಡ)",
    "ml": "Malayalam (മലയാളം)",
    "pa": "Punjabi (ਪੰਜਾਬੀ)",
    "or": "Odia (ଓଡ଼ିଆ)",
    "as": "Assamese (অসমীয়া)",
    "ar": "Arabic",
    "zh-cn": "Chinese (Simplified)",
    "zh-tw": "Chinese (Traditional)",
    "fr": "French",
    "de": "German",
    "es": "Spanish",
    "pt": "Portuguese",
    "ru": "Russian",
    "ja": "Japanese",
    "ko": "Korean",
}


def detect_language(text: str) -> Optional[str]:
    """
    Detect the language of `text`.
    Returns an ISO 639-1 language code (e.g. "en", "hi") or None on failure.
    """
    if not text or len(text.strip()) < 10:
        return None
    try:
        from langdetect import detect
        code = detect(text)
        return code
    except Exception as exc:
        logger.warning(f"Language detection failed: {exc}")
        return None


def get_language_name(code: Optional[str]) -> str:
    """Return a human-readable language name for a given ISO code."""
    if not code:
        return "Unknown"
    return LANGUAGE_NAMES.get(code.lower(), f"Language code: {code}")
