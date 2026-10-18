# One image = API + built frontend. Free host: Render (render.yaml; Hugging Face's Docker SDK is paid-only since 2026). Port 7860.
# Runtime has no torch and no training stack: numpy, scipy, lightgbm, pandas, fastapi, uvicorn, httpx only (CLAUDE.md rule 3).

# ---- stage 1: build the frontend ----------------------------------------------------------------------------------
FROM node:22-slim AS web
WORKDIR /web
COPY frontend/package.json frontend/package-lock.json ./
RUN npm ci
COPY frontend/ ./
RUN npm run build

# ---- stage 2: API + static frontend --------------------------------------------------------------------------------
FROM python:3.12-slim
ENV PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1 PIP_NO_CACHE_DIR=1 \
    AEROGRID_BACKGROUND=1 AEROGRID_MODELS=/app/backend/models SCORECARD_DB=/app/backend/state/scorecard.db FRONTEND_DIST=/app/frontend/dist
# LightGBM needs the OpenMP runtime.
RUN apt-get update && apt-get install -y --no-install-recommends libgomp1 && rm -rf /var/lib/apt/lists/* \
    && useradd --create-home --uid 1000 app
RUN pip install "numpy>=1.26" "scipy>=1.11" "pandas>=2.1" "lightgbm>=4.3" "fastapi>=0.110" "uvicorn>=0.29" "httpx>=0.27"
WORKDIR /app/backend
COPY backend/aerogrid ./aerogrid
COPY backend/data/airports_seed.csv ./data/airports_seed.csv
COPY backend/models ./models
COPY --from=web /web/dist /app/frontend/dist
RUN mkdir -p /app/backend/state && chown -R app:app /app/backend/state
USER app
EXPOSE 7860
HEALTHCHECK --interval=60s --timeout=8s --start-period=40s CMD python -c "import urllib.request; urllib.request.urlopen('http://127.0.0.1:7860/api/health', timeout=6)"
# One worker on purpose: the TTL caches, the flight feed and the scorecard ticker live in process memory.
CMD ["python", "-m", "uvicorn", "aerogrid.api.main:app", "--host", "0.0.0.0", "--port", "7860", "--workers", "1"]
