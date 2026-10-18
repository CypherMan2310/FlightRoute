"""Live ADS-B positions. TTL 15 s. Provider chain: adsb.lol -> adsb.fi -> (opt-in) OpenSky -> simulated.

* adsb.lol: keyless, ODbL 1.0, point/radius query (radius clamped to 250 nm). Default provider.
* adsb.fi (added 2026-10-18): keyless fallback, same JSON as adsb.lol, v3 point query up to 250 nm. Terms (github.com/adsbfi/opendata,
  checked 2026-10-18): personal, non-commercial use only; cite adsb.fi and link its home page (payloads carry `attribution`, the UI
  shows the link); 1 request per second on public endpoints, and requests answered 400/401/403/404/429 count toward that limit while many
  of them cause a temporary IP ban, so calls are spaced one second apart and a provider that answered 429 or 403 is left alone for a while.
* OpenSky: OFF unless OPENSKY_ENABLED=1. Its terms (checked 2026-10-17) limit the REST API to non-profit research
  and education and require a written licence for any operational use (live product, automated system), so a
  deployed AEROGRID must not enable it without one. Credentials are optional: OPENSKY_CLIENT_ID /
  OPENSKY_CLIENT_SECRET use the OAuth2 client-credentials flow (anonymous quota: 400 credits/day).
* airplanes.live is not used: its REST API answered 403 "contact us" to keyless requests (2026-10-17).
"""
from __future__ import annotations

import math
import os
import threading
import time
from typing import Callable

import httpx

from .. import data
from . import simulate
from .cache import UpstreamUnavailable, default_cache, describe_error, fetch_one, http_get, http_post

ADSBLOL_URL = "https://api.adsb.lol/v2/point/{lat}/{lon}/{radius}"
ADSBFI_URL = "https://opendata.adsb.fi/api/v3/lat/{lat}/lon/{lon}/dist/{radius}"
ADSBFI_ATTRIBUTION = "Flight data from adsb.fi (https://adsb.fi), personal non-commercial use"
FI_MIN_INTERVAL_S = 1.0  # adsb.fi public endpoints: 1 request per second
PAUSE_429_S, PAUSE_403_S = 60, 300  # how long a provider is left alone after it answered 429 (unless Retry-After says more) or 403
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


def _guarded_get(name: str, cache, url: str, client):
    """GET that leaves a provider alone after it answered 429 or 403 (Retry-After if given, else 60 s / 300 s): no hammering."""
    hit = cache.get(f"adsb:paused:{name}")
    if hit:
        raise UpstreamUnavailable(f"paused_after_http_{hit[0]}")
    try:
        return http_get(url, client=client)
    except httpx.HTTPStatusError as exc:
        code = exc.response.status_code
        if code in (429, 403):
            retry = exc.response.headers.get("Retry-After", "")
            cache.set(f"adsb:paused:{name}", code, min(3600, int(retry)) if retry.isdigit() else (PAUSE_429_S if code == 429 else PAUSE_403_S))
        raise


def _exchange_flights(rows) -> list[dict]:
    """Aircraft rows in the ADSBexchange v2 schema (adsb.lol and adsb.fi answer with the same fields)."""
    out = []
    for a in rows:
        if a.get("lat") is None or a.get("lon") is None:
            continue
        ground = a.get("alt_baro") == "ground"
        alt = 0 if ground else a.get("alt_baro")
        out.append(_flight(a["hex"], a.get("flight"), a["lat"], a["lon"], alt, ground, a.get("gs"), a.get("track"), a.get("baro_rate"), a.get("t")))
    return out


def _adsblol(lat, lon, radius_nm, client, cache, env) -> list[dict]:
    return _exchange_flights(_guarded_get("adsb.lol", cache, ADSBLOL_URL.format(lat=f"{lat:.2f}", lon=f"{lon:.2f}", radius=f"{radius_nm:g}"), client).json()["ac"])


_fi_lock, _fi_last = threading.Lock(), [0.0]


def _adsbfi(lat, lon, radius_nm, client, cache, env) -> list[dict]:
    with _fi_lock:  # one request per second, also across threads
        wait = _fi_last[0] + FI_MIN_INTERVAL_S - time.monotonic()
        if wait > 0:
            time.sleep(wait)
        _fi_last[0] = time.monotonic()
    return _exchange_flights(_guarded_get("adsb.fi", cache, ADSBFI_URL.format(lat=f"{lat:.2f}", lon=f"{lon:.2f}", radius=f"{radius_nm:g}"), client).json()["ac"])


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
    providers: list[tuple[str, Callable]] = [("adsb.lol", _adsblol), ("adsb.fi", _adsbfi)]
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
            return {"provider": name, "center": {"lat": lat, "lon": lon}, "radius_nm": radius, "count": len(flights), "flights": flights,
                    "primary_failure": ";".join(errors) or None, **({"attribution": ADSBFI_ATTRIBUTION} if name == "adsb.fi" else {})}
        raise UpstreamUnavailable(";".join(errors))

    key = f"adsb:{lat:.2f},{lon:.2f},{radius:g}"
    out = fetch_one(cache, key, ttl=TTL_S, fetch=fetch, simulate=lambda: simulate.sim_flights(lat, lon, radius, cache.clock()))
    if out.get("primary_failure") and not out.get("fallback_reason"):
        out["fallback_reason"] = out["primary_failure"]  # a real answer from a later provider still says why the first one was not used
    return out
