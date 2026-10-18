"""eBird recent observations near a point (api.ebird.org/v2/data/obs/geo/recent). TTL 24 h. Optional key.

Terms (checked 2026-10-17, ebird API Terms of Use, last edited 2021-10-19): a key tied to your own eBird account
(env EBIRD_API_KEY; never share it), non-commercial use, attribute eBird.org wherever the data is shown.
No key -> simulated. Endpoint and parameter names (lat, lng, dist <= 50 km, back <= 30 days, maxResults) are
from eBird API 2.0 as remembered: the Postman docs page would not render, so verify against a real key.
"""
from __future__ import annotations

import os

from . import simulate
from .cache import UpstreamUnavailable, default_cache, fetch_one, http_get

URL = "https://api.ebird.org/v2/data/obs/geo/recent"
PROVIDER = "ebird"
ATTRIBUTION = "eBird.org (Cornell Lab of Ornithology)"
TTL_S = 86400
MAX_DIST_KM = 50
MAX_BACK_DAYS = 30
MAX_RESULTS = 2000
TOP_N = 5


def _summarise(rows: list[dict]) -> dict:
    by_species: dict[str, dict] = {}
    for r in rows:
        s = by_species.setdefault(r["speciesCode"], {"code": r["speciesCode"], "name": r["comName"], "individuals": 0})
        s["individuals"] += int(r["howMany"]) if r.get("howMany") is not None else 1  # presence-only ("X") counts as 1
    top = sorted(by_species.values(), key=lambda s: (-s["individuals"], s["code"]))
    return {"n_records": len(rows), "n_species": len(by_species), "individuals": sum(s["individuals"] for s in top), "top_species": top[:TOP_N]}


def get_recent_observations(lat: float, lng: float, dist_km: float = 25, back_days: int = 14, *, client=None, cache=None, env=None) -> dict:
    if not (-90 <= lat <= 90 and -180 <= lng <= 180):
        raise ValueError("invalid coordinates")
    cache = cache if cache is not None else default_cache()
    env = os.environ if env is None else env
    dist = int(max(1, min(dist_km, MAX_DIST_KM)))
    back = int(max(1, min(back_days, MAX_BACK_DAYS)))

    def fetch() -> dict:
        key = env.get("EBIRD_API_KEY")
        if not key:
            raise UpstreamUnavailable("no_api_key")
        params = {"lat": f"{lat:.2f}", "lng": f"{lng:.2f}", "dist": dist, "back": back, "maxResults": MAX_RESULTS}
        rows = http_get(URL, params=params, headers={"X-eBirdApiToken": key}, client=client).json()
        if not isinstance(rows, list):
            raise ValueError("expected a JSON array")
        return {"provider": PROVIDER, "attribution": ATTRIBUTION, "lat": lat, "lng": lng, "dist_km": dist, "back_days": back, **_summarise(rows)}

    return fetch_one(
        cache,
        f"ebird:{lat:.2f},{lng:.2f},{dist},{back}",
        ttl=TTL_S,
        fetch=fetch,
        simulate=lambda: simulate.sim_ebird(lat, lng, cache.clock(), dist, back),
    )
