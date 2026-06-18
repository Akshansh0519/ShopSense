import os
import logging
from contextlib import asynccontextmanager

from dotenv import load_dotenv
# Load environment variables from .env file before importing local modules
load_dotenv()

from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from slowapi import Limiter, _rate_limit_exceeded_handler
from slowapi.util import get_remote_address
from slowapi.errors import RateLimitExceeded

from app.api.routes import router

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
# Read allowed origins from env. Falls back to localhost only.
_raw_origins = os.getenv("ALLOWED_ORIGINS", "http://localhost:3000,http://localhost:8501")
ALLOWED_ORIGINS = [o.strip() for o in _raw_origins.split(",") if o.strip()]

# ── App Factory ───────────────────────────────────────────────────────────────
app = FastAPI(
    title="ShopSense Recommendation API",
    description="Production-style recommendation serving API",
    version="1.0.0",
    # Disable interactive docs in production to reduce attack surface.
    docs_url=None if IS_PRODUCTION else "/docs",
    redoc_url=None if IS_PRODUCTION else "/redoc",
    openapi_url=None if IS_PRODUCTION else "/openapi.json",
)

# Attach rate limiter state and its default 429 handler
app.state.limiter = limiter
app.add_exception_handler(RateLimitExceeded, _rate_limit_exceeded_handler)

app.add_middleware(
    CORSMiddleware,
    # Restrict to explicit origins — never "*" 
    allow_origins=ALLOWED_ORIGINS,
    allow_credentials=True,
    allow_methods=["GET", "POST"],
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
    if IS_PRODUCTION:
        logger.info("Swagger UI is DISABLED (production mode)")

@app.on_event("shutdown")
async def shutdown_event():
    logger.info("ShopSense API shutting down...")
