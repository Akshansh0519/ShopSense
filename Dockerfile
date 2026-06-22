FROM python:3.11-slim

# ── Security: create a non-root user ─────────────────────────────────────────
# Running as root inside a container is a critical security risk.
# If an attacker exploits the app, they get root on the host via container escape.
RUN addgroup --system appgroup && adduser --system --ingroup appgroup appuser

WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

# Ensure artifacts directory exists and is owned by the app user
RUN mkdir -p artifacts && chown -R appuser:appgroup /app

# ── Drop to non-root user ─────────────────────────────────────────────────────
USER appuser

EXPOSE 8000

# Use --no-access-log in production to avoid leaking user IDs in logs
CMD uvicorn app.main:app --host 0.0.0.0 --port ${PORT:-8000}
