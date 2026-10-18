"""FastAPI app per CLAUDE.md 1.5: JSON API under /api, SSE stream, and the built frontend (frontend/dist) at /.

Every payload says where it came from (`source`: live | cached | simulated) and which model produced it (`trained_on`, kind labels).
No upstream failure can crash a request: connectors degrade to cache, then to labelled simulation.
"""
from __future__ import annotations

import asyncio
import json
import math
import os
import time
from contextlib import asynccontextmanager
from datetime import datetime, timezone
from pathlib import Path
from typing import Callable

import numpy as np
import pandas as pd
from anyio import to_thread
from fastapi import FastAPI, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
from fastapi.staticfiles import StaticFiles

from .. import data
from ..ml import bird, explain, routing, whatif
from ..ml import congestion as cg
from ..ml import ensemble as en
from ..ml import weather as wx
from ..sources import adsb, ebird, metar, metno, nas, openmeteo
from ..sources.cache import TTLCache, combine_sources, default_cache
from .schemas import PredictRequest, WhatIfRequest
from .scorecard import Scorecard

VERSION = "0.1.0"
BACKEND_DIR = Path(__file__).resolve().parents[2]
MAX_AHEAD_H = 36  # forecasts are fetched for 48 h; a flight plus its block time must fit inside
FORECAST_HOURS = 48
US_CODES = {"US", "PR"}
TICK_S, FEED_S = 900, 15
CAT = {0.0: "VFR", 1.0: "MVFR", 2.0: "IFR", 3.0: "LIFR"}
CAUSE_LABELS = {"weather": "Weather", "congestion": "Airport congestion", "cascade": "Cascading delays", "route": "Route and winds", "bird": "Bird activity (prior)", "carrier": "Carrier"}
FEED_HUBS = ["JFK", "ATL", "ORD", "DFW", "LAX", "LHR", "FRA", "DXB", "DEL", "HND", "SIN", "SYD", "GRU", "JNB"]
DATA_SOURCES = [
    {"name": "aviationweather.gov", "role": "METAR / TAF observations", "ttl_s": metar.TTL_S, "license": "US Government work (NOAA / NWS)", "key": "metar"},
    {"name": "Open-Meteo", "role": "hourly forecast, jet-level wind", "ttl_s": openmeteo.TTL_S, "license": "CC-BY 4.0, free non-commercial tier (attribute Open-Meteo.com)", "key": "forecast"},
    {"name": "MET Norway", "role": "forecast fallback when Open-Meteo is unavailable (no gusts, visibility, CAPE or jet wind)", "ttl_s": metno.TTL_S, "license": "CC BY 4.0, free, no key (credit MET Norway)", "key": "forecast_fallback"},
    {"name": "adsb.lol", "role": "live ADS-B positions", "ttl_s": adsb.TTL_S, "license": "ODbL 1.0", "key": "flights"},
    {"name": "adsb.fi", "role": "ADS-B fallback when adsb.lol is unavailable or rate-limited", "ttl_s": adsb.TTL_S, "license": "personal non-commercial use only; cite adsb.fi and link adsb.fi; 1 request/s", "key": "flights_fallback"},
    {"name": "FAA NAS Status", "role": "US airport delay programmes (scorecard ground truth)", "ttl_s": nas.TTL_S, "license": "US Government data", "key": "nas"},
    {"name": "eBird", "role": "bird abundance (optional key)", "ttl_s": ebird.TTL_S, "license": "eBird API terms: non-commercial, attribute eBird.org", "key": "ebird"},
    {"name": "OpenSky", "role": "ADS-B fallback", "ttl_s": adsb.TTL_S, "license": "disabled by default: REST API use in a live service needs a written licence", "key": "opensky"},
]


def _r(x, nd=2):
    return None if x is None else round(float(x), nd)


def _badge(payload: dict, **extra) -> dict:
    meta = {k: payload[k] for k in ("degraded", "attribution") if payload.get(k)}
    return {"source": payload["source"], "age_s": _r(payload.get("age_s"), 1), "fallback_reason": payload.get("fallback_reason"), "provider": payload.get("provider"), **meta, **extra}


class Sources:
    """Live connectors behind one object, with a record of the latest status of each (for /api/health and /api/models)."""

    def __init__(self, client=None, cache=None, env=None):
        self.client = client
        self.cache = cache if cache is not None else default_cache()
        self.env = os.environ if env is None else env
        self.last: dict[str, dict] = {}

    def _note(self, key: str, payload: dict, **extra) -> dict:
        self.last[key] = {**_badge(payload, **extra), "at": self.cache.clock()}
        return payload

    def metar_all(self, icaos: list[str]) -> dict:
        parts = [metar.get_metar(icaos[i : i + metar.MAX_IDS], client=self.client, cache=self.cache) for i in range(0, len(icaos), metar.MAX_IDS)]
        merged = {"source": combine_sources(p["source"] for p in parts), "provider": metar.PROVIDER, "age_s": max(p["age_s"] for p in parts),
                  "fallback_reason": next((p["fallback_reason"] for p in parts if p["fallback_reason"]), None),
                  "stations": {k: v for p in parts for k, v in p["stations"].items()}, "missing": [m for p in parts for m in p["missing"]]}
        return self._note("metar", merged, n_stations=len(merged["stations"]), n_missing=len(merged["missing"]))

    def taf(self, icaos):
        return self._note("taf", metar.get_taf(icaos, client=self.client, cache=self.cache))

    def forecast(self, points, hours: int = FORECAST_HOURS) -> dict:
        """Open-Meteo first; points it could not serve (429, budget, outage) and has no cache for go to MET Norway before being simulated."""
        payload = metno.fill_simulated(openmeteo.get_forecast(points, hours=hours, client=self.client, cache=self.cache), points, hours,
                                       client=self.client, cache=self.cache, env=self.env)
        if payload.get("degraded"):
            self._note("forecast_fallback", payload)
        return self._note("forecast", payload)

    def nas(self) -> dict:
        return self._note("nas", nas.get_status(client=self.client, cache=self.cache))

    def flights(self, lat, lon, radius) -> dict:
        payload = adsb.get_flights(lat, lon, radius, client=self.client, cache=self.cache, env=self.env)
        if payload.get("provider") == "adsb.fi":
            self._note("flights_fallback", payload)
        return self._note("flights", payload)

    def ebird(self, lat, lng) -> dict:
        return self._note("ebird", ebird.get_recent_observations(lat, lng, client=self.client, cache=self.cache, env=self.env))


def _hour_index(hourly: list[dict], ts: float) -> int:
    return int(min(max(round((ts - hourly[0]["time"]) / 3600), 0), len(hourly) - 1))


class World:
    """Builds the live snapshot of every world airport (METAR now, forecast for the target hour) that the models consume."""

    def __init__(self, sources: Sources, clock: Callable[[], float]):
        self.sources, self.clock = sources, clock

    def snapshot(self, target_ts: float, now: float):
        airports = data.load_airports()
        m = self.sources.metar_all([a.icao for a in airports])
        f = self.sources.forecast([(a.lat, a.lon) for a in airports])
        by_icao = m["stations"]
        obs = {a.iata: wx.obs_features(by_icao.get(a.icao)) for a in airports}
        series = {a.iata: p["hourly"] for a, p in zip(airports, f["points"])}
        target_hour = float(math.floor(target_ts / 3600) * 3600)
        fc = {a.iata: wx.fc_features(series[a.iata][_hour_index(series[a.iata], target_hour)]) for a in airports}
        horizon = en.nearest_horizon(max(0.0, (target_ts - now) / 3600))
        info = {"metar": _badge(m, n_stations=len(by_icao), n_missing=len(m["missing"])), "forecast": _badge(f)}
        return {"ts": target_hour, "horizon_h": horizon, "obs": obs, "fc": fc}, series, info


class FlightFeed:
    """Global aircraft view: adsb.lol is point/radius only, so hub regions are refreshed round-robin and merged."""

    def __init__(self, sources: Sources):
        self.sources = sources
        self.latest: dict[str, dict] = {}
        self._next = 0

    def refresh_one(self) -> None:
        code = FEED_HUBS[self._next % len(FEED_HUBS)]
        self._next += 1
        a = data.get_airport(code)
        self.latest[code] = self.sources.flights(a.lat, a.lon, adsb.MAX_RADIUS_NM)

    def snapshot(self, min_regions: int = 2, cap: int = 600) -> dict:
        while len(self.latest) < min_regions:
            self.refresh_one()
        parts = list(self.latest.values())
        seen, flights = set(), []
        for p in parts:
            for fl in p["flights"]:
                if fl["icao24"] not in seen:
                    seen.add(fl["icao24"])
                    flights.append(fl)
        if len(flights) > cap:
            flights = flights[:: math.ceil(len(flights) / cap)]
        real = sorted({p["provider"] for p in parts if p["source"] != "simulated"})  # what the aircraft really came from, not a fixed name
        return {"source": combine_sources(p["source"] for p in parts), "provider": " + ".join(real) or "simulated", "attribution": adsb.ADSBFI_ATTRIBUTION if "adsb.fi" in real else None,
                "fetched_at": min(p["fetched_at"] for p in parts),
                "age_s": max(p["age_s"] for p in parts), "fallback_reason": next((p["fallback_reason"] for p in parts if p["fallback_reason"]), None),
                "regions": len(parts), "count": len(flights), "flights": flights}


def create_app(models_dir=None, db_path=None, sources: Sources | None = None, start_background: bool | None = None, clock: Callable[[], float] = time.time, frontend_dist=None) -> FastAPI:
    models_dir = Path(models_dir or os.environ.get("AEROGRID_MODELS") or BACKEND_DIR / "models")
    db_path = db_path or os.environ.get("SCORECARD_DB") or BACKEND_DIR / "data" / "scorecard.db"
    if start_background is None:
        start_background = os.environ.get("AEROGRID_BACKGROUND", "") == "1"
    sources = sources or Sources()
    bundle = en.Bundle.load_or_none(models_dir)
    bundle_error = None if bundle else en.Bundle.load_error(models_dir)
    scorecard = Scorecard(db_path)
    world, feed = World(sources, clock), FlightFeed(sources)
    airport_by_iata = {a.iata: a for a in data.load_airports()}
    us_airports = [a for a in data.load_airports() if a.country in US_CODES]

    def require_bundle() -> en.Bundle:
        if bundle is None:
            raise HTTPException(503, f"Models not trained or not loadable ({bundle_error}): run `python -m aerogrid.train.train_all` (see TODO.md).")
        return bundle

    def lookup(code: str):
        try:
            return data.get_airport(code)
        except KeyError:
            raise HTTPException(404, f"Unknown airport {code!r}: only the {len(airport_by_iata)} airports in /api/airports are supported.") from None

    def nas_for(payload: dict, a) -> dict:
        if a.country not in US_CODES:
            return {"status": "not covered", "note": "FAA NAS Status covers US airports only"}
        return {k: v for k, v in nas.airport_status(payload, a.iata).items() if k in {"status", "delay_minutes", "ground_stop", "ground_delay"}}

    @asynccontextmanager
    async def lifespan(app: FastAPI):
        tasks = []
        if start_background:
            async def ticker():
                while True:
                    try:
                        await to_thread.run_sync(app.state.tick)
                    except Exception:  # noqa: BLE001 - the loop must survive any single failure
                        pass
                    await asyncio.sleep(TICK_S)

            async def feeder():
                while True:
                    try:
                        await to_thread.run_sync(feed.refresh_one)
                    except Exception:  # noqa: BLE001
                        pass
                    await asyncio.sleep(FEED_S)

            tasks = [asyncio.create_task(ticker()), asyncio.create_task(feeder())]
        yield
        for t in tasks:
            t.cancel()

    app = FastAPI(title="AEROGRID API", version=VERSION, lifespan=lifespan)
    app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["GET", "POST", "OPTIONS"], allow_headers=["*"])
    app.state.bundle, app.state.sources, app.state.scorecard = bundle, sources, scorecard

    # ---- scorecard tick: log predictions for US airports, resolve past ones against live FAA data --------------------
    def tick() -> dict:
        now = clock()
        nas_payload = sources.nas()
        resolved = scorecard.resolve(nas_payload, now)
        logged = 0
        if bundle is not None:
            target = float(math.floor(now / 3600) * 3600 + 2 * 3600)
            snap, _, info = world.snapshot(target, now)
            if info["metar"]["source"] != "simulated" and info["forecast"]["source"] != "simulated" and not info["forecast"].get("degraded"):  # never score the app on simulated or degraded inputs
                score = bundle.score_world(bundle.world_frame(snap))
                idx = {c: i for i, c in enumerate(score["codes"])}
                at_issue = nas_payload["source"] == "live"
                rows = [{"airport": a.iata, "issued_at": now, "target_at": target, "horizon_h": snap["horizon_h"], "pred_dep_min": float(score["gcn"][idx[a.iata], 0]),
                         "nas_event_at_issue": (nas.airport_status(nas_payload, a.iata)["status"] in {"delay", "ground_delay", "ground_stop"}) if at_issue else None,
                         "trained_on": bundle.trained_on} for a in us_airports]
                logged = scorecard.log(rows)
        return {"logged": logged, "resolved": resolved, "nas_source": nas_payload["source"]}

    app.state.tick = tick

    # ---- meta ----------------------------------------------------------------------------------------------------
    @app.get("/api/health")
    def health():
        models = {"status": bundle.status, "trained_on": bundle.trained_on, "created_at": bundle.card.get("created_at")} if bundle else {"status": "untrained", "trained_on": None, "reason": bundle_error}
        return {"status": "ok", "time": clock(), "models": models, "sources": dict(sources.last), "version": VERSION}

    @app.get("/api/models")
    def models():
        if bundle:
            card = bundle.card
        else:
            try:
                card = json.loads((models_dir / "model_card.json").read_text(encoding="utf-8"))
            except (OSError, ValueError):
                card = {"status": "untrained"}
        srcs = [{**{k: v for k, v in s.items() if k != "key"}, "last": sources.last.get(s["key"]), "enabled": s["key"] != "opensky" or str(sources.env.get("OPENSKY_ENABLED", "")) in {"1", "true"}} for s in DATA_SOURCES]
        return {"trained": bundle is not None, "card": card, "data_sources": srcs,
                "badges": {"routing": routing.LABEL, "bird": bird.LABEL, "whatif": whatif.LABEL, "trained_on": card.get("trained_on"),
                           "ground_truth": "FAA NAS Status, US airports only"}}

    @app.get("/api/airports")
    def airports(q: str = "", limit: int = Query(20, ge=1, le=200)):
        q = q.strip().lower()
        month = datetime.fromtimestamp(clock(), tz=timezone.utc).month

        def rank(a):
            if not q:
                return 0
            hay = (a.iata.lower(), a.icao.lower(), a.city.lower(), a.name.lower(), a.country.lower())
            if q in hay[:2]:
                return 0
            if hay[0].startswith(q) or hay[1].startswith(q):
                return 1
            if hay[2].startswith(q) or hay[3].startswith(q):
                return 2
            return 3 if any(q in h for h in hay) else None

        scored = sorted(((rank(a), a.tier, a.iata, a) for a in data.load_airports() if rank(a) is not None), key=lambda t: t[:3])
        return [{"iata": a.iata, "icao": a.icao, "name": a.name, "city": a.city, "country": a.country, "lat": a.lat, "lon": a.lon, "tier": a.tier,
                 "zone": data.weather_zone(a.lat, a.lon, month), "coverage": "us" if a.country in US_CODES else "world (lower confidence)"} for *_, a in scored[:limit]]

    # ---- live data -----------------------------------------------------------------------------------------------
    @app.get("/api/flights")
    def flights(lat: float | None = Query(None, ge=-90, le=90), lon: float | None = Query(None, ge=-180, le=180), radius: float = Query(100, gt=0, le=250)):
        if (lat is None) != (lon is None):
            raise HTTPException(422, "give both lat and lon, or neither for the global view")
        return sources.flights(lat, lon, radius) if lat is not None else feed.snapshot()

    def congestion_payload(limit: int) -> dict:
        b = require_bundle()
        now = clock()
        snap, _, info = world.snapshot(math.floor(now / 3600) * 3600, now)
        raw = b.world_frame(snap)
        score = b.score_world(raw)
        nas_payload = sources.nas()
        rows = []
        for i, a in enumerate(data.load_airports()):
            rows.append({
                "iata": a.iata, "icao": a.icao, "name": a.name, "city": a.city, "country": a.country, "lat": a.lat, "lon": a.lon, "tier": a.tier,
                "stress": _r(score["stress"][i], 1), "pred_dep_delay_min": _r(score["gcn"][i, 0], 1), "pred_nas_min": _r(score["cong_pred"][i], 1),
                "weather_category": CAT.get(snap["obs"][a.iata]["obs_cat"], "VFR"), "nas": nas_for(nas_payload, a),
                "confidence": "normal" if a.country in US_CODES else "low",
            })
        rows.sort(key=lambda r: -r["stress"])
        srcs = {**info, "nas": _badge(nas_payload)}
        return {"source": combine_sources(s["source"] for s in srcs.values()), "fetched_at": now, "trained_on": b.trained_on, "horizon_h": snap["horizon_h"],
                "note": "stress = percentile of the predicted airport-hour delay among training hours (0-100); the 2-hour-horizon model is applied to current conditions",
                "sources": srcs, "airports": rows[:limit]}

    @app.get("/api/congestion")
    def congestion(limit: int = Query(30, ge=1, le=155)):
        return congestion_payload(limit)

    # ---- prediction ----------------------------------------------------------------------------------------------
    def bird_pair(o, d, dep_ts, arr_ts):
        eo, ed = sources.ebird(o.lat, o.lon), sources.ebird(d.lat, d.lon)
        ro, rd = bird.bird_risk(o.lat, o.lon, dep_ts, ebird=eo), bird.bird_risk(d.lat, d.lon, arr_ts, ebird=ed)
        return {"index": _r((ro["risk_index"] + rd["risk_index"]) / 2, 3), "origin": ro, "destination": rd, "ebird": _badge(eo)}

    def advice_lines(o, d, plan, wind, br) -> list[str]:
        lines = []
        if plan["feasible"]:
            if plan["stops"]:
                lines.append(f"Great-circle distance {plan['direct_km']:.0f} km exceeds the usable range of this aircraft class ({plan['usable_range_km']:.0f} km): route via {', '.join(plan['stops'])} (+{plan['detour_pct']}% distance).")
            else:
                lines.append(f"Direct flight within range: {plan['direct_km']:.0f} km great-circle.")
        else:
            lines.append("No chain of legs within this aircraft class's range connects these airports in the seed network.")
        if wind["delta_min"] >= 5:
            lines.append(f"Jet-stream headwinds add about {wind['delta_min']:.0f} min on this route.")
        elif wind["delta_min"] <= -5:
            lines.append(f"Jet-stream tailwinds save about {-wind['delta_min']:.0f} min on this route.")
        for tag, a, r in (("departure", o, br["origin"]), ("arrival", d, br["destination"])):
            lo, hi = r["altitude_band_ft"]
            lines.append(f"Bird migration near {a.iata} at {tag} ({r['label']}): activity {lo:,}-{hi:,} ft AGL, strike-risk index {r['risk_index']:.2f} of 1; climb or descend through this band without level-offs.")
        return lines

    def flight_context(o, d, dep_ts: float, now: float):
        """Everything needed to score one flight: world snapshot, scores, en-route weather, the raw flight row."""
        b = require_bundle()
        snap, series, info = world.snapshot(dep_ts, now)
        score = b.score_world(b.world_frame(snap))
        dist = data.great_circle_km(o.lat, o.lon, d.lat, d.lon)
        crs = b.scheduled_minutes(dist)
        arr_ts = dep_ts + crs * 60
        pts = routing.enroute_points(o.lat, o.lon, d.lat, d.lon)
        ef = sources.forecast(pts)
        fracs = (0.25, 0.5, 0.75)
        er_hours = [ef["points"][k]["hourly"][_hour_index(ef["points"][k]["hourly"], dep_ts + crs * 60 * f)] for k, f in enumerate(fracs)]
        er_fc = [wx.fc_features(h) for h in er_hours]
        er_wind = [(h["wind250_kt"], h["wind250_dir_deg"]) for h in er_hours]
        d_hour = series[d.iata][_hour_index(series[d.iata], arr_ts)]
        return b, snap, info, score, arr_ts, ef, er_fc, er_wind, wx.fc_features(d_hour)

    @app.post("/api/predict")
    def predict(req: PredictRequest):
        b = require_bundle()
        o, d = lookup(req.origin), lookup(req.destination)
        if o.iata == d.iata:
            raise HTTPException(422, "origin and destination must differ")
        now = clock()
        dep_ts = (req.departure_time.replace(tzinfo=req.departure_time.tzinfo or timezone.utc).timestamp()) if req.departure_time else now + 2 * 3600
        if dep_ts < now - 3600:
            raise HTTPException(422, "departure time is in the past")
        if dep_ts > now + MAX_AHEAD_H * 3600:
            raise HTTPException(422, f"departure must be within {MAX_AHEAD_H} h (forecast window)")
        dep_ts = max(dep_ts, now)
        b, snap, info, score, arr_ts, ef, er_fc, er_wind, d_fc = flight_context(o, d, dep_ts, now)
        row = en.build_flight_row(b, o.iata, d.iata, req.airline or "UNK", dep_ts, snap["horizon_h"], snap["obs"][o.iata], snap["obs"][d.iata], snap["fc"][o.iata], d_fc, er_fc, er_wind)
        out = b.predict_flights(row, score)
        total, lo, hi, base = float(out["total"][0]), float(out["lo"][0]), float(out["hi"][0]), float(out["baseline"][0])
        causes = [{"key": k, "label": CAUSE_LABELS[k], "minutes": _r(out["causes"].iloc[0][k]), "share": _r(out["causes"].iloc[0][k] / total if total > 0 else 0, 3)} for k in en.COMPONENTS]
        causes.sort(key=lambda c: -c["minutes"])
        oi = int(out["origin_index"][0])
        wind = routing.wind_extra_minutes(o.lat, o.lon, d.lat, d.lon, er_wind)
        plan = routing.plan_route(o.iata, d.iata, req.aircraft)
        br = bird_pair(o, d, dep_ts, arr_ts)
        nas_payload = sources.nas()
        zone_info = b.ensemble.breakdown(out["components"].iloc[0], row.loc[0, "o_zone"])
        reduced = sorted(set(info["forecast"].get("degraded", [])) | set(_badge(ef).get("degraded", [])))
        drivers = {
            "weather": [{**x, "value": _r(x["value"]), "contribution_min": _r(x["contribution_min"])} for x in explain.lgbm_drivers(b.weather, out["flight_features"], 3)[0]],
            "congestion": [{**x, "value": _r(x["value"]), "contribution_min": _r(x["contribution_min"])} for x in explain.lgbm_drivers(b.congestion.gbm, score["features"].iloc[[oi]], 3)[0]],
            "cascade": [{**x, "value": _r(x["value"]), "contribution_min": _r(x["contribution_min"])} for x in explain.gcn_drivers(b.gcn, en.world_operator(), score["features"][cg.GCN_FEATURES].to_numpy(dtype=float), oi, 1, cg.GCN_FEATURES, 3)],
            "route": [{"label": "jet-stream wind along the route", "contribution_min": _r(wind["delta_min"]),
                       "note": "jet-level wind is unavailable from the fallback forecast: shown as 0" if "wind250" in reduced else "negative = tailwind saves time; only headwind counts as delay"}],
            "bird": [{"label": f"season, local solar time and flyway ({br['origin']['flyway']})", "value": br["index"], "note": bird.LABEL}],
            "carrier": [{"label": (lambda e: f"historical mean for {req.airline or 'all carriers'} ({e['level']}, n={e['n']})")(b.carrier.explain(req.airline or "UNK", o.iata, dep_ts)),
                         "contribution_min": _r(b.carrier.explain(req.airline or "UNK", o.iata, dep_ts)["mean_min"])}],
        }
        us_pair = o.country in US_CODES and d.country in US_CODES
        notes = []
        if not us_pair:
            notes.append("Outside the US the models are lower-confidence: they were trained on US airports only, and FAA status is not available there.")
        if b.trained_on == "synthetic":
            notes.append("These models were trained on SYNTHETIC data for demonstration: the numbers are not real accuracy.")
        if zone_info["pooled_weights"]:
            notes.append(f"Few training rows in the {row.loc[0, 'o_zone']} weather zone: pooled ensemble weights were used.")
        if any(s["source"] == "simulated" for s in (info["metar"], info["forecast"])):
            notes.append("Some weather inputs are simulated because the upstream was unavailable: see sources.")
        if reduced:
            notes.append("Open-Meteo was unavailable, so part of the forecast comes from MET Norway, which has no wind gusts, visibility, CAPE or jet-level wind. "
                         f"Those inputs ({', '.join(reduced)}) are neutral stand-ins, so the weather and route parts are less reliable.")
        card_models = b.card["models"]
        return {
            "origin": o.iata, "destination": d.iata, "airline": req.airline, "departure_time": datetime.fromtimestamp(dep_ts, tz=timezone.utc).isoformat(timespec="minutes"),
            "horizon_h": snap["horizon_h"], "delay_min": _r(total), "interval80": [_r(lo), _r(hi)], "baseline_min": _r(base), "causes": causes, "drivers": drivers,
            "route": {**plan, "wind": {k: _r(v) for k, v in wind.items()}},
            "bird": {"index": br["index"], "origin": br["origin"], "destination": br["destination"], "label": bird.LABEL},
            "advice": advice_lines(o, d, plan, wind, br), "nas": {"origin": nas_for(nas_payload, o), "destination": nas_for(nas_payload, d)},
            "sources": {**info, "enroute_forecast": _badge(ef), "nas": _badge(nas_payload), "ebird": br["ebird"]},
            "models": {k: {"kind": v["kind"], "trained_on": v["trained_on"], **({"label": v["label"]} if "label" in v else {})} for k, v in card_models.items()},
            "confidence": {"level": "normal" if us_pair and not reduced else "low", "notes": notes, "interval": "80% split-conformal (calibrated on a held-out block)"},
            "trained_on": b.trained_on,
        }

    @app.get("/api/advisor")
    def advisor(origin: str, destination: str, aircraft: str = "narrowbody", departure_time: datetime | None = None):
        o, d = lookup(origin), lookup(destination)
        if o.iata == d.iata:
            raise HTTPException(422, "origin and destination must differ")
        if aircraft not in routing.AIRCRAFT:
            raise HTTPException(422, f"aircraft must be one of {sorted(routing.AIRCRAFT)}")
        now = clock()
        dep_ts = departure_time.replace(tzinfo=departure_time.tzinfo or timezone.utc).timestamp() if departure_time else now + 2 * 3600
        dep_ts = min(max(dep_ts, now), now + MAX_AHEAD_H * 3600)
        dist = data.great_circle_km(o.lat, o.lon, d.lat, d.lon)
        crs = (bundle.scheduled_minutes(dist) if bundle else dist / 13.3 + 28)
        arr_ts = dep_ts + crs * 60
        ef = sources.forecast(routing.enroute_points(o.lat, o.lon, d.lat, d.lon))
        hours = [ef["points"][k]["hourly"][_hour_index(ef["points"][k]["hourly"], dep_ts + crs * 60 * f)] for k, f in enumerate((0.25, 0.5, 0.75))]
        wind = routing.wind_extra_minutes(o.lat, o.lon, d.lat, d.lon, [(h["wind250_kt"], h["wind250_dir_deg"]) for h in hours])
        plan = routing.plan_route(o.iata, d.iata, aircraft)
        br = bird_pair(o, d, dep_ts, arr_ts)
        advice = advice_lines(o, d, plan, wind, br)
        if "wind250" in ef.get("degraded", []):
            advice.append("Jet-level wind is unavailable (forecast fallback: MET Norway), so wind effects on this route are not included.")
        return {"route": plan, "wind": {k: _r(v) for k, v in wind.items()}, "bird": {"index": br["index"], "origin": br["origin"], "destination": br["destination"], "label": bird.LABEL},
                "advice": advice, "sources": {"enroute_forecast": _badge(ef), "ebird": br["ebird"]}, "departure_time": datetime.fromtimestamp(dep_ts, tz=timezone.utc).isoformat(timespec="minutes")}

    @app.post("/api/whatif")
    def whatif_endpoint(req: WhatIfRequest):
        b = require_bundle()
        a = lookup(req.airport)
        now = clock()
        snap, _, info = world.snapshot(math.floor(now / 3600) * 3600, now)
        res = whatif.run(b, b.world_frame(snap), a.iata, req.kind, req.severity)
        fmt = lambda d: {k: (_r(v, 2) if isinstance(v, float) else v) for k, v in d.items()}  # noqa: E731
        res["deltas"], res["affected"] = [fmt(d) for d in res["deltas"]], [fmt(d) for d in res["affected"]]
        return {**res, "sources": info}

    @app.get("/api/scorecard")
    def scorecard_endpoint():
        return {**scorecard.metrics(clock()), "trained_on": bundle.trained_on if bundle else None, "scored_airports": len(us_airports), "model_horizon_h": 2,
                "note": "Predictions are logged only when METAR and forecast inputs are real (live or cached) and complete (not the reduced MET Norway fallback), and are resolved only against live FAA data."}

    @app.get("/api/stream")
    async def stream(max_events: int | None = Query(None, ge=1), interval_s: float = Query(FEED_S, ge=0, le=120)):
        async def gen():
            sent = 0
            while True:
                snap = await to_thread.run_sync(feed.snapshot)
                yield f"event: flights\ndata: {json.dumps(snap)}\n\n"
                sent += 1
                if max_events and sent >= max_events:
                    return
                if bundle is not None:
                    top = await to_thread.run_sync(congestion_payload, 20)
                    yield f"event: congestion\ndata: {json.dumps(top)}\n\n"
                else:
                    yield ": no trained models, congestion events disabled\n\n"
                sent += 1
                if max_events and sent >= max_events:
                    return
                await asyncio.sleep(interval_s)

        return StreamingResponse(gen(), media_type="text/event-stream", headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"})

    dist = Path(frontend_dist or os.environ.get("FRONTEND_DIST") or BACKEND_DIR.parent / "frontend" / "dist")
    if dist.is_dir():
        app.mount("/", StaticFiles(directory=dist, html=True), name="frontend")
    return app


app = create_app()
