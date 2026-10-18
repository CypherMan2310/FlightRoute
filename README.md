---
title: AEROGRID
sdk: docker
app_port: 7860
pinned: false
short_description: Flight delay prediction and airport congestion on a live globe
---

# AEROGRID

Predicts flight delay and its cause, and monitors airport congestion on a black-and-white live globe. A college project (AERO-SIT, Phase-I review), so every number says where it came from: each API payload carries `source` = live | cached | simulated, and each model card says `trained_on` = real | synthetic.

Pages: Globe, Search (delay, 80% interval, cause breakdown, route and altitude advice), Congestion, What-if, Scorecard (the app's own accuracy against FAA NAS delay programs), Models (model cards, baselines, data freshness).

## Honest status
The bundle in `backend/models` is a **reduced real run**: 4 months of US data (2026-04..07), 42 airports, a 10-day test block. The congestion model beats its per-airport hour-of-week baseline by 2.6% RMSE (narrowly); weather delay and the carrier lookup are not distinguishable from a constant; non-US outputs are low-confidence (derived routes graph). The 30-month run (`train.yml`) is still to do. Details and every variant tried: `docs/phase2-design.md`, sections 7 and 8; open work: `TODO.md`.

## Run it
```bash
cd frontend && npm ci && npm run build          # the UI
cd ../backend && pip install -r requirements.txt
AEROGRID_BACKGROUND=1 python -m uvicorn aerogrid.api.main:app --port 7860   # API + built UI on :7860
```
Tests: `cd backend && python -m pytest` (offline). Docker: `docker build -t aerogrid . && docker run -p 7860:7860 aerogrid` (one image, API + built frontend, runtime has no torch).

Optional environment variables: `EBIRD_API_KEY` (otherwise bird abundance is simulated and labelled so), `OPENSKY_ENABLED` (off by default: its terms need a licence for a live service). Never commit keys.

## Deploy (free, no card): Hugging Face Spaces
1. Create a Space at huggingface.co/new-space: SDK **Docker**, hardware CPU basic (free), any name.
2. Push this repository to it (the `README.md` front matter above is what Spaces reads):
   ```bash
   git remote add space https://huggingface.co/spaces/<user>/<space>
   git push space main      # username = your HF user, password = an HF access token with write access
   ```
3. Optional: in the Space settings add the secret `EBIRD_API_KEY`; in this GitHub repository set the variable `AEROGRID_URL` to `https://<user>-<space>.hf.space` so `ingest.yml` keeps the app awake and archives the scorecard (a free Space's disk is ephemeral, so the scorecard database resets on restart).

Data and attribution: adsb.lol (ODbL 1.0), Open-Meteo.com (CC-BY 4.0), eBird.org, aviationweather.gov / NOAA, FAA NAS Status, BTS On-Time Performance, IEM ASOS.
