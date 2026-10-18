"""Live ADS-B positions. TTL 15 s. Provider chain: adsb.lol -> (opt-in) OpenSky -> simulated.

* adsb.lol: keyless, ODbL 1.0, point/radius query (radius clamped to 250 nm). Default provider.
* OpenSky: OFF unless OPENSKY_ENABLED=1. Its terms (checked 2026-10-17) limit the REST API to non-profit research
  and education and require a written licence for any operational use (live product, automated system), so a
  deployed AEROGRID must not enable it without one. Credentials are optional: OPENSKY_CLIENT_ID /
  OPENSKY_CLIENT_SECRET use the OAuth2 client-credentials flow (anonymous quota: 400 credits/day).
* airplanes.live is not used: its REST API answered 403 "contact us" to keyless requests (2026-10-17).
"""
from __future__ import annotations

import math
import os
from typing import Callable

import httpx

from .. import data
from . import simulate
from .cache import UpstreamUnavailable, default_cache, describe_error, fetch_one, http_get, http_post

ADSBLOL_URL = "https://api.adsb.lol/v2/point/{lat}/{lon}/{radius}"
OPENSKY_STATES_URL = "https://opensky-network.org/api/states/all"
OPENSKY_TOKEN_URL = "https://auth.opensky-network.org/auth/realms/opensky-network/protocol/openid-connect/token"
TTL_S = 15
MAX_RADIUS_NM = 250
TOKEN_TTL_S = 1500  # OpenSky tokens live 30 min
_KM_PER_NM = 1.852
_FT_PER_M = 1 / 0.3048
_KT_PER_MS = 1.943844
_FPM_PER_MS = 196.850394


def _flight(icao24, callsign, lat, lon, alt_ft, on_ground, gs_kt, track_deg, vrate_fpm, aircraft_type) -> dict:
    return {
        "icao24": icao24,
        "callsign": (callsign or "").strip() or None,
        "lat": lat,
        "lon": lon,
        "alt_ft": alt_ft,
        "on_ground": on_ground,
        "gs_kt": gs_kt,
        "track_deg": track_deg,
        "vrate_fpm": vrate_fpm,
        "aircraft_type": aircraft_type,
    }


def _adsblol(lat, lon, radius_nm, client, cache, env) -> list[dict]:
    resp = http_get(ADSBLOL_URL.format(lat=f"{lat:.2f}", lon=f"{lon:.2f}", radius=f"{radius_nm:g}"), client=client)
    out = []
    for a in resp.json()["ac"]:
        if a.get("lat") is None or a.get("lon") is None:
            continue
        ground = a.get("alt_baro") == "ground"
        alt = 0 if ground else a.get("alt_baro")
        out.append(_flight(a["hex"], a.get("flight"), a["lat"], a["lon"], alt, ground, a.get("gs"), a.get("track"), a.get("baro_rate"), a.get("t")))
    return out


def _opensky_token(client_id: str, client_secret: str, client, cache) -> str:
    hit = cache.get("opensky:token")
    if hit:
        return hit[0]
    resp = http_post(
        OPENSKY_TOKEN_URL,
        data={"grant_type": "client_credentials", "client_id": client_id, "client_secret": client_secret},
        client=client,
    )
    token = resp.json()["access_token"]
    cache.set("opensky:token", token, TOKEN_TTL_S)
    return token


def _opensky(lat, lon, radius_nm, client, cache, env) -> list[dict]:
    if cache.get("opensky:blocked"):
        raise UpstreamUnavailable("paused")
    headers = {}
    cid, secret = env.get("OPENSKY_CLIENT_ID"), env.get("OPENSKY_CLIENT_SECRET")
    if cid and secret:
        headers["Authorization"] = f"Bearer {_opensky_token(cid, secret, client, cache)}"
    dlat = radius_nm / 60
    dlon = dlat / max(0.05, math.cos(math.radians(lat)))
    params = {"lamin": lat - dlat, "lomin": lon - dlon, "lamax": lat + dlat, "lomax": lon + dlon}
    try:
        resp = http_get(OPENSKY_STATES_URL, params=params, headers=headers, client=client)
    except httpx.HTTPStatusError as exc:
        if exc.response.status_code == 429:
            retry = exc.response.headers.get("X-Rate-Limit-Retry-After-Seconds", "60")
            cache.set("opensky:blocked", True, min(3600, int(retry)) if retry.isdigit() else 60)
        raise
    out = []
    for s in resp.json().get("states") or []:
        if s[5] is None or s[6] is None:
            continue
        if data.great_circle_km(lat, lon, s[6], s[5]) > radius_nm * _KM_PER_NM:
            continue
        alt_m = s[7] if s[7] is not None else s[13]
        alt = 0 if s[8] else (round(alt_m * _FT_PER_M) if alt_m is not None else None)
        out.append(
            _flight(
                s[0], s[1], s[6], s[5], alt, bool(s[8]),
                round(s[9] * _KT_PER_MS, 1) if s[9] is not None else None,
                s[10],
                round(s[11] * _FPM_PER_MS) if s[11] is not None else None,
                None,
            )
        )
    return out


def get_flights(lat: float, lon: float, radius_nm: float = 100, *, client=None, cache=None, env=None) -> dict:
    """Aircraft within `radius_nm` of (lat, lon). Payload: source/provider/fetched_at/age_s, center, radius_nm,
    count, flights[icao24, callsign, lat, lon, alt_ft, on_ground, gs_kt, track_deg, vrate_fpm, aircraft_type]."""
    if not (-90 <= lat <= 90 and -180 <= lon <= 180) or not radius_nm > 0:
        raise ValueError("invalid centre or radius")
    cache = cache if cache is not None else default_cache()
    env = os.environ if env is None else env
    radius = min(float(radius_nm), MAX_RADIUS_NM)
    providers: list[tuple[str, Callable]] = [("adsb.lol", _adsblol)]
    if str(env.get("OPENSKY_ENABLED", "")).lower() in {"1", "true", "yes"}:
        providers.append(("opensky", _opensky))

    def fetch() -> dict:
        errors = []
        for name, call in providers:
            try:
                flights = call(lat, lon, radius, client, cache, env)
            except Exception as exc:  # noqa: BLE001 - try the next provider
                errors.append(f"{name}:{describe_error(exc)}")
                continue
            return {"provider": name, "center": {"lat": lat, "lon": lon}, "radius_nm": radius, "count": len(flights), "flights": flights}
        raise UpstreamUnavailable(";".join(errors))

    key = f"adsb:{lat:.2f},{lon:.2f},{radius:g}"
    return fetch_one(cache, key, ttl=TTL_S, fetch=fetch, simulate=lambda: simulate.sim_flights(lat, lon, radius, cache.clock()))
