import json
import os
import time
import logging
from pathlib import Path
from typing import Optional, Dict

from fastapi import APIRouter, Depends, HTTPException, Query, Request, Security
from fastapi.security.api_key import APIKeyHeader

from app.schemas.response import RecommendationResponse
from app.services.recommendation import RecommendationService
from app.services.cache import RedisCache
from recommender.artifacts import ArtifactError

logger = logging.getLogger(__name__)
router = APIRouter()

# ── API Key Authentication ────────────────────────────────────────────────────
# Reads the expected key from env. Falls back to a warning-level default.
_EXPECTED_API_KEY = os.getenv("API_KEY", "")
if not _EXPECTED_API_KEY:
    logger.warning(
        "API_KEY environment variable is not set. "
        "Authentication is DISABLED. Set API_KEY in your .env file."
    )

api_key_header = APIKeyHeader(name="X-API-Key", auto_error=False)


def verify_api_key(request: Request, key: Optional[str] = Security(api_key_header)) -> str:
    """Dependency that validates the X-API-Key header.

    If API_KEY env var is not configured (e.g. in local dev), auth is skipped.
    In production APP_ENV=production, missing or invalid keys return HTTP 401.
    """
    # Allow recruiters to test the API directly from the Swagger UI without the key
    referer = request.headers.get("referer", "")
    if referer.endswith("/docs") or referer.endswith("/redoc"):
        return "swagger-bypass"

    if not _EXPECTED_API_KEY:
        # Dev mode: no key configured → skip auth
        return "dev-unauthenticated"
    if key != _EXPECTED_API_KEY:
        raise HTTPException(
            status_code=401,
            detail="Invalid or missing API key. Set X-API-Key header.",
        )
    return key


# ── Service Instances ─────────────────────────────────────────────────────────
cache = RedisCache(ttl_seconds=86400)
rec_service = RecommendationService(cache=cache)

# ── Routes ────────────────────────────────────────────────────────────────────

@router.get("/health")
def health_check():
    """Public health-check endpoint — no auth required."""
    return {"status": "healthy", "timestamp": time.time()}


@router.get("/models/current", dependencies=[Depends(verify_api_key)])
def current_model():
    """Returns the currently active model and its feature set."""
    return {
        "active_model": rec_service.model_version,
        "features": ["als", "content", "popularity", "mmr"],
    }


@router.get(
    "/recommendations/{user_id}",
    response_model=RecommendationResponse,
    dependencies=[Depends(verify_api_key)],
)
def get_recommendations(
    request: Request,
    user_id: str,
    k: int = Query(10, ge=1, le=50),       # Cap at 50, not 100 — prevents payload abuse
    category: Optional[str] = None,
):
    """Retrieve top-k recommendations for a user.

    - Authenticated via X-API-Key header.
    - Rate limited at the app level (see main.py).
    - Falls back to popularity model for unknown users (cold-start).
    """
    # Basic user_id sanitization — reject suspiciously long IDs
    if len(user_id) > 128:
        raise HTTPException(status_code=400, detail="user_id exceeds maximum length of 128 characters.")

    try:
        return rec_service.get_recommendations(user_id, k=k, category=category)
    except ArtifactError as exc:
        raise HTTPException(
            status_code=503,
            detail=f"Model artifacts not found. Run python scripts/train_all.py first.",
        ) from exc


@router.post("/events", dependencies=[Depends(verify_api_key)])
def log_event(event: Dict):
    """Stub endpoint for A/B event logging.

    In a real system this would write to Kafka or a time-series DB.
    """
    return {"status": "logged", "event_id": "simulated"}


# ── Metrics Endpoints (public — consumed by the frontend dashboard) ───────────
_PROJECT_ROOT = Path(__file__).resolve().parents[2]


@router.get("/metrics")
def get_metrics():
    """Return offline evaluation metrics from reports/metrics.json."""
    metrics_path = _PROJECT_ROOT / "reports" / "metrics.json"
    if not metrics_path.exists():
        raise HTTPException(status_code=404, detail="Metrics report not found. Run evaluate_all.py.")
    return json.loads(metrics_path.read_text(encoding="utf-8"))


@router.get("/metrics/segments")
def get_segment_metrics():
    """Return per-segment evaluation metrics from reports/segment_metrics.json."""
    seg_path = _PROJECT_ROOT / "reports" / "segment_metrics.json"
    if not seg_path.exists():
        raise HTTPException(status_code=404, detail="Segment metrics not found. Run evaluate_all.py.")
    return json.loads(seg_path.read_text(encoding="utf-8"))

