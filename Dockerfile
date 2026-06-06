# ── Stage 1: Builder ──────────────────────────────────────────────────────────
# Install dependencies in a separate stage to keep the final image lean
FROM python:3.11-slim AS builder

WORKDIR /app

# Install build tools needed for some Python packages (e.g. shap, numpy)
RUN apt-get update && apt-get install -y --no-install-recommends \
    gcc \
    g++ \
    && rm -rf /var/lib/apt/lists/*

# Copy and install dependencies first (layer caching — only rebuilds if requirements change)
COPY requirements.txt .
RUN pip install --upgrade pip && \
    pip install --prefix=/install --no-cache-dir -r requirements.txt


# ── Stage 2: Runtime ──────────────────────────────────────────────────────────
FROM python:3.11-slim

WORKDIR /app

# Copy installed packages from builder
COPY --from=builder /install /usr/local

# Copy application code
COPY Main.py .
COPY model/ ./model/

# Non-root user for security (best practice in production containers)
RUN useradd --create-home --shell /bin/bash appuser
USER appuser

# Expose the FastAPI port
EXPOSE 8000

# Health check — Docker will mark the container unhealthy if this fails
HEALTHCHECK --interval=30s --timeout=10s --start-period=15s --retries=3 \
    CMD python -c "import urllib.request; urllib.request.urlopen('http://localhost:8000/health')"

# Run with uvicorn — single worker for now, scale with --workers in production
CMD ["python", "-m", "uvicorn", "Main:app", "--host", "0.0.0.0", "--port", "8000"]