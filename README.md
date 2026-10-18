# AEROGRID

Predicts flight delay and its cause, and monitors airport congestion on a black-and-white live globe whose airport bars are coloured by traffic (green free, yellow moderate, red heavy). A college project (AERO-SIT, Phase-I review), so every number says where it came from: each API payload carries `source` = live | cached | simulated, and each model card says `trained_on` = real | synthetic.

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

Optional environment variables: `EBIRD_API_KEY` (otherwise bird abundance is simulated and labelled so), `AEROGRID_CONTACT` (contact string sent to MET Norway, the forecast fallback when Open-Meteo is rate-limited; default the repository URL), `OPENSKY_ENABLED` (off by default: its terms need a licence for a live service). Never commit keys.

## Deploy (free): Render
Hugging Face Spaces no longer offer Docker on free accounts (checked 2026-10-18: the Docker SDK shows "Paid"), so the free path is Render, which runs this repository's Dockerfile as a free web service. `render.yaml` is the blueprint.
1. Sign in at render.com with GitHub and allow it to read this repository (private is fine).
2. **New > Blueprint**, pick this repository and branch `main`, and apply. It creates one free Docker web service named `aerogrid` (it sets `PORT=7860`, matching the Dockerfile).
3. Wait for the first build (about 5 to 10 minutes). The app is then at `https://aerogrid-<suffix>.onrender.com`; `/api/health` should show `"trained_on":"real"`.
4. Optional: add `EBIRD_API_KEY` in the service's Environment tab; set the GitHub repository variable `AEROGRID_URL` to the service URL for `ingest.yml`.

Free-tier limits: 512 MB RAM (the app uses about 175 MB), about 0.1 CPU, the service sleeps after 15 minutes without traffic and takes about a minute to wake, and its disk is ephemeral, so the scorecard database resets on every restart or wake. `ingest.yml` pings every 30 minutes, which does not keep a free instance awake (see TODO.md).

Data and attribution: adsb.lol (ODbL 1.0), adsb.fi (fallback, personal non-commercial use), MET Norway (CC BY 4.0, forecast fallback), Open-Meteo.com (CC-BY 4.0), eBird.org, aviationweather.gov / NOAA, FAA NAS Status, BTS On-Time Performance, IEM ASOS.
