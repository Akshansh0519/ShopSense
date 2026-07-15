import os
import logging

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
# Uses the client's IP address as the rate-limiting key.
limiter = Limiter(key_func=get_remote_address)

# ── Environment ──────────────────────────────────────────────────────────────
APP_ENV = os.getenv("APP_ENV", "development")
IS_PRODUCTION = APP_ENV == "production"

# ── CORS ─────────────────────────────────────────────────────────────────────
# Read allowed origins from env. Falls back to a permissive wildcard so the
# API works even if the env var is not configured on the host.
# IMPORTANT: In production, set ALLOWED_ORIGINS in Render's env dashboard to:
#   https://shop-sense-five.vercel.app,https://shopsense-pkys.onrender.com
_raw_origins = os.getenv("ALLOWED_ORIGINS", "*")
ALLOWED_ORIGINS = [o.strip() for o in _raw_origins.split(",") if o.strip()]

# Per the CORS spec, allow_credentials=True is INCOMPATIBLE with allow_origins=["*"].
# Browsers will silently block all responses if both are set. Detect and handle this.
_is_wildcard = ALLOWED_ORIGINS == ["*"]

# ── App Factory ───────────────────────────────────────────────────────────────
app = FastAPI(
    title="ShopSense Recommendation API",
    description="Production-style recommendation serving API",
    version="1.0.0",
    # Enable interactive docs so recruiters can test the API directly.
    docs_url="/docs",
    redoc_url="/redoc",
    openapi_url="/openapi.json",
)

# Attach rate limiter state and its default 429 handler
app.state.limiter = limiter
app.add_exception_handler(RateLimitExceeded, _rate_limit_exceeded_handler)

app.add_middleware(
    CORSMiddleware,
    allow_origins=ALLOWED_ORIGINS,
    allow_credentials=not _is_wildcard,  # Must be False when origins is ["*"]
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

@app.on_event("startup")
async def startup_event():
    logger.info("ShopSense API starting up in %s mode...", APP_ENV)

    # Pre-load all ML model artifacts into RAM BEFORE Uvicorn accepts any
    # HTTP traffic. This blocks the port from opening until models are ready,
    # so /health only returns 200 OK when the system is TRULY ready to serve.
    # Consequence: first request after cold-start wakeup responds in < 50ms.
    logger.info("Pre-loading ML models into memory (may take 30-60s on cold start)...")
    rec_service.load_artifacts()
    logger.info("All ML artifacts loaded. API is now ready to serve requests.")

@app.on_event("shutdown")
async def shutdown_event():
    logger.info("ShopSense API shutting down...")
