"""Open-Meteo forecast (api.open-meteo.com/v1/forecast). TTL 30 min, cached per point (rounded to 0.01 deg).

Free tier (checked 2026-10-17): non-commercial use only, < 10,000 calls/day, 5,000/hour, 600/minute, no key,
data CC-BY 4.0 (attribute Open-Meteo.com wherever it is displayed). One request carries all points.
The archive endpoint is for training (train/real.py) and is not wrapped here.
"""
from __future__ import annotations

from datetime import datetime, timezone

from . import simulate
from .cache import UpstreamUnavailable, batch_fetch, default_cache, http_get

URL = "https://api.open-meteo.com/v1/forecast"
PROVIDER = "open-meteo.com"
TTL_S = 1800
MAX_HOURS = 168
DAILY_BUDGET = 8000  # points per UTC day; the free limit is 10,000 calls/day, keep headroom for the other users of the key-less pool
_FIELDS = {
    "temperature_2m": "temp_c",
    "precipitation": "precip_mm",
    "wind_speed_10m": "wind_kt",
    "wind_gusts_10m": "gust_kt",
    "visibility": "visibility_m",
    "cloud_cover_low": "cloud_low_pct",
    "cape": "cape_jkg",
    "weather_code": "weather_code",
    "wind_speed_250hPa": "wind250_kt",  # jet-level wind, used by routing
    "wind_direction_250hPa": "wind250_dir_deg",
}
_HOURLY = tuple(_FIELDS)


def _epoch(iso_utc: str) -> int:
    return int(datetime.fromisoformat(iso_utc).replace(tzinfo=timezone.utc).timestamp())


def _normalise(row: dict) -> dict:
    h = row["hourly"]
    times = h["time"]
    hourly = [{"time": _epoch(t), **{out: h[src][i] for src, out in _FIELDS.items()}} for i, t in enumerate(times)]
    return {"hourly": hourly}


def get_forecast(points, hours: int = 24, *, client=None, cache=None) -> dict:
    """Hourly forecast for each (lat, lon). Payload: source/provider/fetched_at/age_s, hours, points[{lat, lon, source, hourly[...]}]."""
    points = [(float(la), float(lo)) for la, lo in points]
    if not points:
        raise ValueError("at least one point is required")
    if not 1 <= hours <= MAX_HOURS:
        raise ValueError(f"hours must be 1..{MAX_HOURS}")
    cache = cache if cache is not None else default_cache()
    keys = [f"{la:.2f},{lo:.2f}" for la, lo in points]
    ids = list(dict.fromkeys(keys))

    def fetch_many(todo: list[str]) -> dict[str, dict]:
        budget_key = f"openmeteo:budget:{int(cache.clock() // 86400)}"
        used = (cache.get(budget_key) or (0, 0))[0]
        if used + len(todo) > DAILY_BUDGET:
            raise UpstreamUnavailable("daily_budget")
        lats, lons = zip(*(k.split(",") for k in todo))
        params = {
            "latitude": ",".join(lats),
            "longitude": ",".join(lons),
            "hourly": ",".join(_HOURLY),
            "forecast_hours": hours,
            "timezone": "UTC",
            "wind_speed_unit": "kn",
        }
        body = http_get(URL, params=params, client=client).json()
        rows = body if isinstance(body, list) else [body]
        if len(rows) != len(todo):
            raise ValueError("unexpected number of locations in the answer")
        cache.set(budget_key, used + len(todo), 86400)
        return {k: _normalise(r) for k, r in zip(todo, rows)}

    def simulate_one(k: str) -> dict:
        la, lo = map(float, k.split(","))
        return {"hourly": simulate.sim_forecast(la, lo, cache.clock(), hours)["hourly"]}

    batch = batch_fetch(cache, f"openmeteo:{hours}", ids, ttl=TTL_S, fetch_many=fetch_many, simulate_one=simulate_one)
    out = [{**batch.records[k], "lat": la, "lon": lo} for k, (la, lo) in zip(keys, points)]
    return {**batch.meta(PROVIDER), "hours": hours, "points": out}
