"""Route planning and wind-adjusted flight time. ALGORITHMIC (not ML).

* great-circle path and the three en-route sample points (25 / 50 / 75 %),
* range-aware stop insertion: shortest chain of legs, each within the aircraft's usable range, over the seed airports
  (Dijkstra; every extra stop costs a ground-time penalty so stops are only added when needed),
* wind-adjusted time from jet-level (250 hPa) forecast winds at the three en-route points; `extra_min` (headwind cost only,
  >= 0) is the route component fed to the ensemble.
Aircraft classes are generic approximations (typical TAS and still-air range), not type-specific performance data.
"""
from __future__ import annotations

import math
from functools import lru_cache

import numpy as np
from scipy import sparse
from scipy.sparse.csgraph import dijkstra

from .. import data

LABEL = "algorithmic (not ML)"
KM_PER_NM = 1.852
DEFAULT_TAS_KT = 450.0
RANGE_RESERVE = 0.85  # usable fraction of still-air range (reserves, alternates)
GROUND_STOP_MIN = 45
AIRCRAFT = {
    "regional": {"name": "Regional jet (CRJ-900 class)", "tas_kt": 430.0, "range_nm": 1550},
    "narrowbody": {"name": "Narrow-body (A320 / 737 class)", "tas_kt": 450.0, "range_nm": 3000},
    "widebody": {"name": "Wide-body (787 / A350 class)", "tas_kt": 490.0, "range_nm": 7000},
    "ultra_long": {"name": "Ultra-long-range (A350-900ULR class)", "tas_kt": 490.0, "range_nm": 9000},
}
_SEGMENTS = (0.0, 0.375, 0.625, 1.0)  # each en-route wind sample (25/50/75 %) governs one segment


def enroute_points(o_lat: float, o_lon: float, d_lat: float, d_lon: float) -> list[tuple[float, float]]:
    return [data.great_circle_point(o_lat, o_lon, d_lat, d_lon, f) for f in (0.25, 0.5, 0.75)]


def _vec_point(lat1, lon1, lat2, lon2, f):
    """Vectorised data.great_circle_point."""
    p1, l1, p2, l2 = (np.radians(np.asarray(a, dtype=float)) for a in (lat1, lon1, lat2, lon2))
    v1 = np.stack([np.cos(p1) * np.cos(l1), np.cos(p1) * np.sin(l1), np.sin(p1)])
    v2 = np.stack([np.cos(p2) * np.cos(l2), np.cos(p2) * np.sin(l2), np.sin(p2)])
    omega = np.arccos(np.clip((v1 * v2).sum(axis=0), -1, 1))
    s = np.where(np.sin(omega) < 1e-12, 1.0, np.sin(omega))
    a, b = np.sin((1 - f) * omega) / s, np.sin(f * omega) / s
    v = a * v1 + b * v2
    return np.degrees(np.arctan2(v[2], np.hypot(v[0], v[1]))), np.degrees(np.arctan2(v[1], v[0]))


def _vec_bearing(lat1, lon1, lat2, lon2):
    p1, p2, dl = np.radians(lat1), np.radians(lat2), np.radians(np.asarray(lon2) - np.asarray(lon1))
    y = np.sin(dl) * np.cos(p2)
    x = np.cos(p1) * np.sin(p2) - np.sin(p1) * np.cos(p2) * np.cos(dl)
    return np.degrees(np.arctan2(y, x)) % 360


def wind_extra_minutes_vec(o_lat, o_lon, d_lat, d_lon, speed, wind_from, tas_kt: float = DEFAULT_TAS_KT) -> dict:
    """Vectorised over flights. speed / wind_from: (3, n) jet wind (kt, direction the wind blows FROM) at the 25/50/75 % points.
    Headwind component = speed * cos(from - track); each sample governs one segment of the great circle."""
    o_lat, o_lon, d_lat, d_lon = (np.asarray(a, dtype=float) for a in (o_lat, o_lon, d_lat, d_lon))
    p1, p2 = np.radians(o_lat), np.radians(d_lat)
    h = np.sin((p2 - p1) / 2) ** 2 + np.cos(p1) * np.cos(p2) * np.sin(np.radians(d_lon - o_lon) / 2) ** 2
    total_nm = 2 * data.EARTH_RADIUS_KM * np.arcsin(np.minimum(1.0, np.sqrt(h))) / KM_PER_NM
    t_wind = np.zeros_like(total_nm)
    for k, (f0, f1) in enumerate(zip(_SEGMENTS, _SEGMENTS[1:])):
        a, b = _vec_point(o_lat, o_lon, d_lat, d_lon, f0), _vec_point(o_lat, o_lon, d_lat, d_lon, f1)
        track = _vec_bearing(a[0], a[1], b[0], b[1])
        headwind = np.asarray(speed, dtype=float)[k] * np.cos(np.radians(np.asarray(wind_from, dtype=float)[k] - track))
        t_wind += total_nm * (f1 - f0) / np.maximum(100.0, tas_kt - headwind)
    t_wind_min, t_still_min = t_wind * 60, total_nm / tas_kt * 60
    delta = t_wind_min - t_still_min
    return {"nowind_min": t_still_min, "wind_min": t_wind_min, "delta_min": delta, "extra_min": np.maximum(0.0, delta)}


def wind_extra_minutes(o_lat, o_lon, d_lat, d_lon, winds, tas_kt: float = DEFAULT_TAS_KT) -> dict:
    """Scalar form of wind_extra_minutes_vec. winds: three (speed_kt, direction_from_deg) at the 25/50/75 % points."""
    speed = np.array([[w[0]] for w in winds], dtype=float)
    wind_from = np.array([[w[1]] for w in winds], dtype=float)
    out = wind_extra_minutes_vec([o_lat], [o_lon], [d_lat], [d_lon], speed, wind_from, tas_kt)
    return {k: float(v[0]) for k, v in out.items()}


@lru_cache(maxsize=1)
def _dist_km() -> np.ndarray:
    return data._distance_matrix(data.load_airports())


def plan_route(origin: str, dest: str, aircraft: str = "narrowbody") -> dict:
    """Direct if within usable range, otherwise the shortest feasible chain of legs through seed airports."""
    if aircraft not in AIRCRAFT:
        raise ValueError(f"unknown aircraft class {aircraft!r}; choose from {sorted(AIRCRAFT)}")
    o, d = data.get_airport(origin), data.get_airport(dest)
    if o.iata == d.iata:
        raise ValueError("origin and destination are the same airport")
    spec = AIRCRAFT[aircraft]
    usable_km = spec["range_nm"] * RANGE_RESERVE * KM_PER_NM
    codes = [a.iata for a in data.load_airports()]
    dist = _dist_km()
    i, j = codes.index(o.iata), codes.index(d.iata)
    direct = float(dist[i, j])
    base = {"label": LABEL, "aircraft": aircraft, "direct_km": direct, "usable_range_km": usable_km}
    if direct <= usable_km:
        path = [o.iata, d.iata]
    else:
        penalty = GROUND_STOP_MIN / 60 * spec["tas_kt"] * KM_PER_NM
        ok = (dist <= usable_km) & (dist > 0)
        graph = sparse.csr_matrix(np.where(ok, dist + penalty, 0.0))
        _, pred = dijkstra(graph, directed=True, indices=i, return_predecessors=True)
        if pred[j] < 0:
            return {**base, "feasible": False, "path": [], "stops": [], "legs": [], "total_km": None, "detour_pct": None, "time_min_nowind": None}
        idx = [j]
        while idx[-1] != i:
            idx.append(int(pred[idx[-1]]))
        path = [codes[k] for k in reversed(idx)]
    legs = [{"from": a, "to": b, "km": float(dist[codes.index(a), codes.index(b)])} for a, b in zip(path, path[1:])]
    total = sum(leg["km"] for leg in legs)
    stops = path[1:-1]
    return {
        **base, "feasible": True, "path": path, "stops": stops, "legs": legs, "total_km": total,
        "detour_pct": round((total - direct) / direct * 100, 1),
        "time_min_nowind": total / KM_PER_NM / spec["tas_kt"] * 60 + GROUND_STOP_MIN * len(stops),
    }
