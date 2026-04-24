"""
Image Analysis Service
----------------------
Detects potential image manipulation by examining:
  - Error Level Analysis (ELA) — differences in JPEG compression artifacts
  - Metadata inconsistencies (EXIF data)
  - Basic statistical properties (noise, edge density)

Note: For a production system, integrate a dedicated deepfake / GAN-detection
model (e.g. FaceForensics++). This implementation provides a solid
heuristic baseline that runs without GPU requirements.
"""

import io
import math
import logging
from typing import List, Tuple

from PIL import Image, ImageChops, ImageFilter, ImageEnhance

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Error Level Analysis (ELA)
# ---------------------------------------------------------------------------

def _ela_score(image: Image.Image, quality: int = 90) -> float:
    """
    Compute a normalised ELA score.
    Higher values indicate more compression-level inconsistencies,
    which may suggest editing.
    """
    # Save at reduced quality and reload
    buffer = io.BytesIO()
    image.save(buffer, format="JPEG", quality=quality)
    buffer.seek(0)
    compressed = Image.open(buffer)

    # Difference image
    diff = ImageChops.difference(image.convert("RGB"), compressed.convert("RGB"))

    # Amplify and measure mean brightness
    enhanced = ImageEnhance.Brightness(diff).enhance(10)
    pixels = list(enhanced.getdata())

    # Average channel value across pixels
    total = sum(sum(p) for p in pixels)
    avg = total / (len(pixels) * 3) if pixels else 0
    return round(avg, 2)


# ---------------------------------------------------------------------------
# Metadata extraction
# ---------------------------------------------------------------------------

def _extract_metadata(image: Image.Image) -> dict:
    """Return selected EXIF fields if available."""
    try:
        exif_data = image._getexif()  # PIL internal
        if not exif_data:
            return {}
        interesting_tags = {
            271: "Make",
            272: "Model",
            305: "Software",
            306: "DateTime",
            36867: "DateTimeOriginal",
        }
        return {
            label: exif_data[tag]
            for tag, label in interesting_tags.items()
            if tag in exif_data
        }
    except Exception:
        return {}


# ---------------------------------------------------------------------------
# Edge density (high edge count can indicate pasting / stitching)
# ---------------------------------------------------------------------------

def _edge_density(image: Image.Image) -> float:
    """Return fraction of pixels identified as edges."""
    gray = image.convert("L")
    edges = gray.filter(ImageFilter.FIND_EDGES)
    pixels = list(edges.getdata())
    edge_pixels = sum(1 for p in pixels if p > 30)
    return round(edge_pixels / len(pixels), 4) if pixels else 0.0


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def analyze_image(image_bytes: bytes) -> dict:
    """
    Analyse an uploaded image for signs of manipulation.

    Returns: label, confidence, credibility_score, explanation,
             key_phrases, sources
    """
    try:
        image = Image.open(io.BytesIO(image_bytes))
    except Exception as exc:
        logger.error(f"Cannot open image: {exc}")
        return {
            "label": "UNKNOWN",
            "confidence": 0.0,
            "credibility_score": 0,
            "explanation": "Could not process the image file.",
            "key_phrases": [],
            "sources": [],
        }

    ela = _ela_score(image)
    edge_density = _edge_density(image)
    metadata = _extract_metadata(image)

    observations: List[str] = []
    manipulation_score = 0  # higher → more suspicious

    # --- ELA check ---
    if ela > 18:
        manipulation_score += 3
        observations.append(f"High ELA score ({ela:.1f}) — compression inconsistencies detected")
    elif ela > 10:
        manipulation_score += 1
        observations.append(f"Moderate ELA score ({ela:.1f}) — minor inconsistencies")
    else:
        observations.append(f"Low ELA score ({ela:.1f}) — compression looks uniform")

    # --- Edge density check ---
    if edge_density > 0.25:
        manipulation_score += 2
        observations.append(f"High edge density ({edge_density:.2%}) — possible compositing")
    elif edge_density > 0.15:
        manipulation_score += 1
        observations.append(f"Moderate edge density ({edge_density:.2%})")

    # --- Metadata check ---
    if not metadata:
        manipulation_score += 1
        observations.append("No EXIF metadata found — data may have been stripped")
    else:
        if "Software" in metadata:
            software = metadata["Software"]
            editing_tools = ["photoshop", "gimp", "lightroom", "affinity", "pixlr"]
            if any(tool in software.lower() for tool in editing_tools):
                manipulation_score += 2
                observations.append(f"Edited with '{software}' detected in EXIF")
            else:
                observations.append(f"Camera/software: {software}")

        if "DateTimeOriginal" in metadata:
            observations.append(f"Original capture date: {metadata['DateTimeOriginal']}")

    # --- Resolution check ---
    width, height = image.size
    if width < 200 or height < 200:
        manipulation_score += 1
        observations.append(f"Very low resolution ({width}×{height}) — may be a screenshot or re-uploaded image")

    # --- Verdict ---
    if manipulation_score >= 5:
        label = "FAKE"
        confidence = min(0.55 + manipulation_score * 0.06, 0.95)
        credibility_score = max(5, 40 - manipulation_score * 5)
        explanation = (
            "Multiple indicators of image manipulation detected. "
            "This image shows significant signs of editing or compositing."
        )
    elif manipulation_score >= 3:
        label = "MISLEADING"
        confidence = min(0.50 + manipulation_score * 0.07, 0.88)
        credibility_score = max(20, 65 - manipulation_score * 7)
        explanation = (
            "Some indicators suggest this image may have been edited. "
            "Treat with caution and verify from original sources."
        )
    else:
        label = "REAL"
        confidence = min(0.55 + (5 - manipulation_score) * 0.06, 0.88)
        credibility_score = min(90, 70 + (5 - manipulation_score) * 4)
        explanation = (
            "No strong signs of manipulation detected. "
            "Image compression and metadata appear consistent."
        )

    return {
        "label": label,
        "confidence": round(confidence, 3),
        "credibility_score": credibility_score,
        "explanation": explanation,
        "key_phrases": observations[:6],
        "sources": [
            "https://fotoforensics.com",
            "https://www.tineye.com",
            "https://images.google.com",
            "https://www.invid-project.eu",
        ],
    }
