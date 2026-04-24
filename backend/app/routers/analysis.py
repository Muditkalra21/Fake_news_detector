"""
Analysis Router
---------------
REST API endpoints for all analysis types:
  POST /api/analyze/text   — news text analysis
  POST /api/analyze/image  — image upload analysis
  POST /api/analyze/video  — video URL analysis
"""

import time
import logging
from fastapi import APIRouter, HTTPException, UploadFile, File
from fastapi.responses import JSONResponse

from app.models.schemas import TextAnalysisRequest, VideoAnalysisRequest, AnalysisResult
from app.services.text_analyzer import analyze_text, get_active_model_info
from app.services.image_analyzer import analyze_image
from app.services.video_analyzer import analyze_video
from app.services.language_detector import detect_language, get_language_name

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/api/analyze", tags=["analysis"])

# Log which model is active when this router is loaded
_model_info = get_active_model_info()
logger.info("=" * 60)
logger.info(f"[ROUTER] Analysis router loaded")
logger.info(f"[ROUTER] Active model : {_model_info['model_id']}")
logger.info(f"[ROUTER] Model choice : {_model_info['choice']} ({_model_info['size']})")
logger.info(f"[ROUTER] Task type    : {_model_info['task']}")
logger.info("=" * 60)


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

    logger.info("-" * 50)
    logger.info(f"[TEXT] New analysis request")
    logger.info(f"[TEXT] Language detected : {lang_code} ({lang_name})")
    logger.info(f"[TEXT] Text length       : {len(request.text)} chars")
    logger.info(f"[TEXT] Text preview      : {request.text[:100].strip()}...")

    t0 = time.perf_counter()
    try:
        result = analyze_text(request.text)
        elapsed = (time.perf_counter() - t0) * 1000

        result["detected_language"] = lang_name

        logger.info(f"[TEXT] ✓ Completed in {elapsed:.0f}ms")
        logger.info(f"[TEXT] Label          : {result.get('label')}")
        logger.info(f"[TEXT] Confidence     : {result.get('confidence', 0) * 100:.1f}%")
        logger.info(f"[TEXT] Credibility    : {result.get('credibility_score')}/100")
        logger.info("-" * 50)

        return JSONResponse(content=result)

    except Exception as exc:
        logger.error(f"[TEXT] ✗ Analysis failed after {(time.perf_counter()-t0)*1000:.0f}ms: {exc}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Text analysis failed: {str(exc)}")


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

    logger.info("-" * 50)
    logger.info(f"[IMAGE] New analysis request")
    logger.info(f"[IMAGE] Filename    : {file.filename}")
    logger.info(f"[IMAGE] Content-Type: {file.content_type}")
    logger.info(f"[IMAGE] Size        : {len(contents) / 1024:.1f} KB")

    t0 = time.perf_counter()
    try:
        result = analyze_image(contents)
        elapsed = (time.perf_counter() - t0) * 1000

        result["detected_language"] = None

        logger.info(f"[IMAGE] ✓ Completed in {elapsed:.0f}ms")
        logger.info(f"[IMAGE] Label      : {result.get('label')}")
        logger.info(f"[IMAGE] Credibility: {result.get('credibility_score')}/100")
        logger.info("-" * 50)

        return JSONResponse(content=result)

    except Exception as exc:
        logger.error(f"[IMAGE] ✗ Analysis failed: {exc}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Image analysis failed: {str(exc)}")


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

    logger.info("-" * 50)
    logger.info(f"[VIDEO] New analysis request")
    logger.info(f"[VIDEO] URL        : {url[:100]}")
    logger.info(f"[VIDEO] Description: {(request.description or '')[:80]}")

    t0 = time.perf_counter()
    try:
        result = analyze_video(url, request.description or "")
        elapsed = (time.perf_counter() - t0) * 1000

        result["detected_language"] = None

        logger.info(f"[VIDEO] ✓ Completed in {elapsed:.0f}ms")
        logger.info(f"[VIDEO] Label      : {result.get('label')}")
        logger.info(f"[VIDEO] Credibility: {result.get('credibility_score')}/100")
        logger.info("-" * 50)

        return JSONResponse(content=result)

    except Exception as exc:
        logger.error(f"[VIDEO] ✗ Analysis failed: {exc}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Video analysis failed: {str(exc)}")

