"""
Analysis Router
---------------
REST API endpoints for all analysis types:
  POST /api/analyze/text   — news text analysis
  POST /api/analyze/image  — image upload analysis
  POST /api/analyze/video  — video URL analysis
"""

import logging
from fastapi import APIRouter, HTTPException, UploadFile, File, Form
from fastapi.responses import JSONResponse

from app.models.schemas import TextAnalysisRequest, VideoAnalysisRequest, AnalysisResult
from app.services.text_analyzer import analyze_text
from app.services.image_analyzer import analyze_image
from app.services.video_analyzer import analyze_video
from app.services.language_detector import detect_language, get_language_name

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/api/analyze", tags=["analysis"])


# ---------------------------------------------------------------------------
# Text analysis endpoint
# ---------------------------------------------------------------------------

@router.post("/text", response_model=AnalysisResult)
async def analyze_text_endpoint(request: TextAnalysisRequest):
    """
    Analyse a news article or social media post for credibility.

    - Classifies as REAL / FAKE / MISLEADING
    - Returns confidence score, credibility score, and explanation
    """
    if not request.text or len(request.text.strip()) < 20:
        raise HTTPException(
            status_code=422,
            detail="Text must be at least 20 characters long."
        )

    # Detect language
    lang_code = detect_language(request.text)
    lang_name = get_language_name(lang_code)
    logger.info(f"Text analysis requested | lang={lang_code} | length={len(request.text)}")

    result = analyze_text(request.text)
    result["detected_language"] = lang_name

    return JSONResponse(content=result)


# ---------------------------------------------------------------------------
# Image analysis endpoint
# ---------------------------------------------------------------------------

@router.post("/image", response_model=AnalysisResult)
async def analyze_image_endpoint(file: UploadFile = File(...)):
    """
    Analyse an uploaded image for signs of manipulation.

    Accepts: JPEG, PNG, WebP, BMP
    """
    # Validate content type
    allowed_types = {"image/jpeg", "image/png", "image/webp", "image/bmp", "image/gif"}
    if file.content_type not in allowed_types:
        raise HTTPException(
            status_code=415,
            detail=f"Unsupported file type: {file.content_type}. Allowed: JPEG, PNG, WebP, BMP"
        )

    # Limit file size to 10 MB
    MAX_SIZE = 10 * 1024 * 1024
    contents = await file.read()
    if len(contents) > MAX_SIZE:
        raise HTTPException(
            status_code=413,
            detail="Image too large. Maximum size is 10 MB."
        )

    logger.info(f"Image analysis requested | filename={file.filename} | size={len(contents)} bytes")

    result = analyze_image(contents)
    result["detected_language"] = None

    return JSONResponse(content=result)


# ---------------------------------------------------------------------------
# Video analysis endpoint
# ---------------------------------------------------------------------------

@router.post("/video", response_model=AnalysisResult)
async def analyze_video_endpoint(request: VideoAnalysisRequest):
    """
    Analyse a video URL for credibility based on source and metadata signals.
    """
    url = request.url.strip()
    if not url.startswith(("http://", "https://")):
        raise HTTPException(
            status_code=422,
            detail="Please provide a valid URL starting with http:// or https://"
        )

    logger.info(f"Video analysis requested | url={url[:80]}")

    result = analyze_video(url, request.description or "")
    result["detected_language"] = None

    return JSONResponse(content=result)
