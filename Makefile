PY ?= python

.PHONY: test run dev build train-synth train-real

test:
	cd backend && $(PY) -m pytest

# API + built frontend on :7860 (same as the Docker image). Needs `make build` first for the UI.
run:
	cd backend && AEROGRID_BACKGROUND=1 $(PY) -m uvicorn aerogrid.api.main:app --host 0.0.0.0 --port 7860

# Two terminals: the API on :8000 and the Vite dev server on :5173 (proxies /api).
dev:
	cd backend && AEROGRID_BACKGROUND=1 $(PY) -m uvicorn aerogrid.api.main:app --reload --port 8000

build:
	cd frontend && npm ci && npm run build

# SYNTHETIC smoke training into a temp dir (never overwrites backend/models). Real training: see train.yml.
train-synth:
	cd backend && $(PY) -m aerogrid.train.train_all --source synthetic --days 45 --airports 20

# Real data (slow, cached under .work): BTS + IEM ASOS + Open-Meteo. Writes backend/models.
train-real:
	cd backend && $(PY) -m aerogrid.train.train_all --source real --months 30 --work ../.work --out models --threads 4
