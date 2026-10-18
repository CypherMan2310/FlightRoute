"""METAR and TAF from the aviationweather.gov Data API (/api/data/metar, /api/data/taf). TTL 10 min.

Upstream limits (checked 2026-10-17): 100 requests/min, custom User-Agent expected, max 400 rows per call.
Stations are ICAO codes (use data.get_airport(iata).icao). A station with no METAR is reported in `missing`,
never invented; a failed upstream falls back per station to stale cache, then simulation.
"""
from __future__ import annotations

from datetime import datetime

from . import simulate
from .cache import batch_fetch, default_cache, http_get

BASE = "https://aviationweather.gov/api/data"
PROVIDER = "aviationweather.gov"
TTL_S = 600
MAX_IDS = 100
_CEILING_COVERS = {"BKN", "OVC", "OVX", "VV"}


def flight_category(visibility_sm: float | None, ceiling_ft: float | None) -> str:
    """FAA flight category from visibility (statute miles) and ceiling (ft AGL): VFR | MVFR | IFR | LIFR."""
    vis = 10.0 if visibility_sm is None else visibility_sm
    ceil = 99999.0 if ceiling_ft is None else ceiling_ft
    if vis < 1 or ceil < 500:
        return "LIFR"
    if vis < 3 or ceil < 1000:
        return "IFR"
    if vis <= 5 or ceil <= 3000:
        return "MVFR"
    return "VFR"


def _visibility(v) -> float | None:
    if v is None:
        return None
    if isinstance(v, str):
        v = v.strip().rstrip("+")
    return float(v)


def _clouds(rows) -> list[dict]:
    return [{"cover": c["cover"], "base_ft": c["base"]} for c in rows or [] if c.get("base") is not None]


def _ceiling(clouds: list[dict]) -> int | None:
    bases = [c["base_ft"] for c in clouds if c["cover"] in _CEILING_COVERS]
    return min(bases) if bases else None


def _wind_dir(v) -> int | None:
    return None if v is None or v == "VRB" else int(v)


def _normalise_metar(r: dict) -> dict:
    clouds = _clouds(r.get("clouds"))
    vis = _visibility(r.get("visib"))
    ceiling = _ceiling(clouds)
    return {
        "icao": r["icaoId"],
        "name": r.get("name"),
        "lat": r.get("lat"),
        "lon": r.get("lon"),
        "obs_time": int(r["obsTime"]),
        "temp_c": r.get("temp"),
        "dewpoint_c": r.get("dewp"),
        "wind_dir_deg": _wind_dir(r.get("wdir")),
        "wind_kt": r.get("wspd"),
        "gust_kt": r.get("wgst"),
        "visibility_sm": vis,
        "altimeter_hpa": r.get("altim"),
        "wx": r.get("wxString") or "",
        "clouds": clouds,
        "ceiling_ft": ceiling,
        "flight_category": r.get("fltCat") or flight_category(vis, ceiling),
        "raw": r.get("rawOb"),
    }


def _normalise_taf(r: dict) -> dict:
    periods = []
    for f in r.get("fcsts") or []:
        clouds = _clouds(f.get("clouds"))
        vis = _visibility(f.get("visib"))
        ceiling = _ceiling(clouds)
        periods.append(
            {
                "from": int(f["timeFrom"]),
                "to": int(f["timeTo"]),
                "change": f.get("fcstChange"),
                "probability": f.get("probability"),
                "wind_dir_deg": _wind_dir(f.get("wdir")),
                "wind_kt": f.get("wspd"),
                "gust_kt": f.get("wgst"),
                "visibility_sm": vis,
                "wx": f.get("wxString") or "",
                "clouds": clouds,
                "ceiling_ft": ceiling,
                "flight_category": flight_category(vis, ceiling),
            }
        )
    return {
        "icao": r["icaoId"],
        "issue_time": int(datetime.fromisoformat(r["issueTime"]).timestamp()),
        "valid_from": int(r["validTimeFrom"]),
        "valid_to": int(r["validTimeTo"]),
        "raw": r.get("rawTAF"),
        "periods": periods,
    }


def _ids(icaos) -> list[str]:
    ids = list(dict.fromkeys(i.strip().upper() for i in icaos if i and i.strip()))
    if not ids:
        raise ValueError("at least one station id is required")
    if len(ids) > MAX_IDS:
        raise ValueError(f"at most {MAX_IDS} stations per call")
    return ids


def _fetch_rows(product: str, ids: list[str], client) -> list[dict]:
    resp = http_get(f"{BASE}/{product}", params={"ids": ",".join(ids), "format": "json"}, client=client)
    if resp.status_code == 204 or not resp.content.strip():
        return []
    rows = resp.json()
    if not isinstance(rows, list):
        raise ValueError("expected a JSON array")
    return rows


def _latest(rows: list[dict], time_key: str, normalise) -> dict[str, dict]:
    best: dict[str, dict] = {}
    for r in rows:
        if r["icaoId"] not in best or r[time_key] > best[r["icaoId"]][time_key]:
            best[r["icaoId"]] = r
    return {k: normalise(v) for k, v in best.items()}


def _run(product, namespace, normalise, time_key, simulator, icaos, client, cache) -> dict:
    cache = cache if cache is not None else default_cache()
    ids = _ids(icaos)
    batch = batch_fetch(
        cache,
        namespace,
        ids,
        ttl=TTL_S,
        fetch_many=lambda todo: _latest(_fetch_rows(product, todo, client), time_key, normalise),
        simulate_one=lambda i: simulator(i, cache.clock()),
    )
    return {**batch.meta(PROVIDER), "stations": batch.records, "missing": batch.missing}


def get_metar(icaos, *, client=None, cache=None) -> dict:
    return _run("metar", "metar", _normalise_metar, "obsTime", simulate.sim_metar, icaos, client, cache)


def get_taf(icaos, *, client=None, cache=None) -> dict:
    return _run("taf", "taf", _normalise_taf, "issueTime", simulate.sim_taf, icaos, client, cache)
