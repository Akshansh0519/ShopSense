import os
import logging
import threading

from dotenv import load_dotenv
# Load environment variables from .env file before importing local modules
load_dotenv()

from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from slowapi import Limiter, _rate_limit_exceeded_handler
from slowapi.util import get_remote_address
from slowapi.errors import RateLimitExceeded

from app.api.routes import router, rec_service

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)

# ── Rate Limiter ─────────────────────────────────────────────────────────────
limiter = Limiter(key_func=get_remote_address)

# ── Environment ──────────────────────────────────────────────────────────────
APP_ENV = os.getenv("APP_ENV", "development")
IS_PRODUCTION = APP_ENV == "production"

# ── CORS ─────────────────────────────────────────────────────────────────────
# Defaults to "*" so the API works even if ALLOWED_ORIGINS is not set in
# the host's environment dashboard (e.g. a fresh Render deployment).
# For production hardening, set ALLOWED_ORIGINS in Render's env vars to:
#   https://shop-sense-five.vercel.app,https://<your-render-slug>.onrender.com
_raw_origins = os.getenv("ALLOWED_ORIGINS", "*")
ALLOWED_ORIGINS = [o.strip() for o in _raw_origins.split(",") if o.strip()]
_is_wildcard = ALLOWED_ORIGINS == ["*"]

# ── App Factory ───────────────────────────────────────────────────────────────
app = FastAPI(
    title="ShopSense Recommendation API",
    description="Production-style recommendation serving API",
    version="1.0.0",
    docs_url="/docs",
    redoc_url="/redoc",
    openapi_url="/openapi.json",
)

app.state.limiter = limiter
app.add_exception_handler(RateLimitExceeded, _rate_limit_exceeded_handler)

app.add_middleware(
    CORSMiddleware,
    allow_origins=ALLOWED_ORIGINS,
    allow_credentials=not _is_wildcard,
    allow_methods=["GET", "POST", "OPTIONS"],
    allow_headers=["Authorization", "Content-Type", "X-API-Key"],
)

# ── Security Headers Middleware ───────────────────────────────────────────────
@app.middleware("http")
async def add_security_headers(request: Request, call_next):
    response = await call_next(request)
    response.headers["X-Content-Type-Options"] = "nosniff"
    response.headers["X-Frame-Options"] = "DENY"
    response.headers["X-XSS-Protection"] = "1; mode=block"
    response.headers["Referrer-Policy"] = "strict-origin-when-cross-origin"
    response.headers["Cache-Control"] = "no-store"
    return response

app.include_router(router)

# ── Startup: background model loading ────────────────────────────────────────
# IMPORTANT: We load models in a background thread, NOT blocking the startup
# hook. This lets Uvicorn open the port immediately (satisfying Render's
# port-open timeout), while models warm up in parallel.
# Use GET /ready to poll whether models are fully loaded.
_models_loaded = False

def _load_models_background():
    global _models_loaded
    logger.info("Background thread: Pre-loading ML models into memory...")
    rec_service.load_artifacts()
    _models_loaded = True
    logger.info("Background thread: All ML artifacts loaded. API is ready to serve.")

@app.on_event("startup")
async def startup_event():
    logger.info("ShopSense API starting up in %s mode...", APP_ENV)
    # Fire model loading in a daemon thread — port opens immediately
    t = threading.Thread(target=_load_models_background, daemon=True)
    t.start()

@app.on_event("shutdown")
async def shutdown_event():
    logger.info("ShopSense API shutting down...")

# ── /ready endpoint ───────────────────────────────────────────────────────────
@app.get("/ready")
def readiness_check():
    """
    Returns 200 OK only when ML models are fully loaded into memory.
    Returns 503 while models are still warming up in the background.
    Use this endpoint (not /health) to poll for true readiness.
    """
    if _models_loaded:
        return {"status": "ready", "models_loaded": True}
    return JSONResponse(
        status_code=503,
        content={"status": "loading", "models_loaded": False,
                 "message": "ML models are warming up. Please wait ~60s and retry."}
    )
