"""Airports, routes graph and weather zones."""
from __future__ import annotations

import csv
import math
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path

import numpy as np
from scipy import sparse

SEED_PATH = Path(__file__).resolve().parents[1] / "data" / "airports_seed.csv"
EARTH_RADIUS_KM = 6371.0088
ZONES = ("tropical", "subtropical", "temperate", "subpolar", "monsoon")


@dataclass(frozen=True)
class Airport:
    iata: str
    icao: str
    name: str
    city: str
    country: str
    lat: float
    lon: float
    tier: int  # 1 = global mega-hub, 2 = major hub, 3 = large regional (hand-assigned)
    notes: str


@lru_cache(maxsize=4)
def _load(path: str) -> tuple[Airport, ...]:
    with open(path, newline="", encoding="utf-8") as f:
        return tuple(
            Airport(r["iata"], r["icao"], r["name"], r["city"], r["country"], float(r["lat"]), float(r["lon"]), int(r["tier"]), r["notes"])
            for r in csv.DictReader(f)
        )


def load_airports(path: Path | str = SEED_PATH) -> tuple[Airport, ...]:
    return _load(str(path))


def get_airport(code: str) -> Airport:
    """Look up by IATA or ICAO code (case-insensitive). Raises KeyError if unknown."""
    code = code.strip().upper()
    for a in load_airports():
        if code in (a.iata, a.icao):
            return a
    raise KeyError(code)


# ---- geometry -------------------------------------------------------------------------------------------------

def great_circle_km(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    p1, p2 = math.radians(lat1), math.radians(lat2)
    h = math.sin((p2 - p1) / 2) ** 2 + math.cos(p1) * math.cos(p2) * math.sin(math.radians(lon2 - lon1) / 2) ** 2
    return 2 * EARTH_RADIUS_KM * math.asin(min(1.0, math.sqrt(h)))


def initial_bearing(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    """Initial true bearing in degrees [0, 360) from point 1 towards point 2."""
    p1, p2, dl = math.radians(lat1), math.radians(lat2), math.radians(lon2 - lon1)
    y = math.sin(dl) * math.cos(p2)
    x = math.cos(p1) * math.sin(p2) - math.sin(p1) * math.cos(p2) * math.cos(dl)
    return math.degrees(math.atan2(y, x)) % 360


def great_circle_point(lat1: float, lon1: float, lat2: float, lon2: float, f: float) -> tuple[float, float]:
    """Point at fraction f (0..1) along the great circle from (lat1, lon1) to (lat2, lon2)."""
    p1, l1, p2, l2 = map(math.radians, (lat1, lon1, lat2, lon2))
    v1 = (math.cos(p1) * math.cos(l1), math.cos(p1) * math.sin(l1), math.sin(p1))
    v2 = (math.cos(p2) * math.cos(l2), math.cos(p2) * math.sin(l2), math.sin(p2))
    omega = math.acos(max(-1.0, min(1.0, sum(a * b for a, b in zip(v1, v2)))))
    s = math.sin(omega)
    if s < 1e-12:
        return lat1, lon1
    a, b = math.sin((1 - f) * omega) / s, math.sin(f * omega) / s
    x, y, z = (a * p + b * q for p, q in zip(v1, v2))
    return math.degrees(math.atan2(z, math.hypot(x, y))), math.degrees(math.atan2(y, x))


# ---- weather zones --------------------------------------------------------------------------------------------

# Asian summer-monsoon domains: (lat_min, lat_max, lon_min, lon_max, first_month, last_month).
# Southern-hemisphere and African monsoons are deliberately not modelled: they fall back to the latitude bands.
_MONSOON_BOXES = (
    (0.0, 33.0, 68.0, 93.0, 6, 9),  # South Asia, SW monsoon
    (5.0, 32.0, 93.0, 125.0, 5, 10),  # Indochina, southern China, Taiwan, Philippines
)
_BANDS = ((23.5, "tropical"), (35.0, "subtropical"), (58.0, "temperate"))


def weather_zone(lat: float, lon: float, month: int) -> str:
    """tropical | subtropical | temperate | subpolar | monsoon.

    Coarse latitude bands (23.5 / 35 / 58 deg) plus a seasonal Asian-monsoon override. A modelling convenience
    for per-zone ensemble weights, not a climate classification. `month` only matters inside a monsoon box.
    """
    if not (-90 <= lat <= 90 and -180 <= lon <= 180):
        raise ValueError(f"invalid coordinates: {lat}, {lon}")
    if not (isinstance(month, int) and 1 <= month <= 12):
        raise ValueError(f"month must be 1..12, got {month!r}")
    for la0, la1, lo0, lo1, m0, m1 in _MONSOON_BOXES:
        if la0 <= lat <= la1 and lo0 <= lon <= lo1 and m0 <= month <= m1:
            return "monsoon"
    for limit, zone in _BANDS:
        if abs(lat) < limit:
            return zone
    return "subpolar"


# ---- routes graph ---------------------------------------------------------------------------------------------

@dataclass(frozen=True, eq=False)
class RoutesGraph:
    """Undirected airport graph. `adjacency[i, j]` = great-circle km if a route exists, else 0.

    `origin` says where the edges came from: "derived" = modelled from the seed tiers (NOT real schedules),
    anything else names the real source passed to `build_routes_graph`.
    """

    codes: tuple[str, ...]
    index: dict[str, int]
    adjacency: sparse.csr_matrix
    origin: str

    def neighbors(self, code: str) -> list[str]:
        row = self.adjacency.getrow(self.index[code])
        return sorted(self.codes[j] for j in row.indices)

    def degree(self, code: str) -> int:
        return int(self.adjacency.getrow(self.index[code]).nnz)

    def distance_km(self, a: str, b: str) -> float:
        pa, pb = get_airport(a), get_airport(b)
        return great_circle_km(pa.lat, pa.lon, pb.lat, pb.lon)


def _distance_matrix(airports: tuple[Airport, ...]) -> np.ndarray:
    lat = np.radians([a.lat for a in airports])
    lon = np.radians([a.lon for a in airports])
    dlat = lat[:, None] - lat[None, :]
    dlon = lon[:, None] - lon[None, :]
    h = np.sin(dlat / 2) ** 2 + np.cos(lat)[:, None] * np.cos(lat)[None, :] * np.sin(dlon / 2) ** 2
    return 2 * EARTH_RADIUS_KM * np.arcsin(np.minimum(1.0, np.sqrt(h)))


def _derived_pairs(airports: tuple[Airport, ...], dist: np.ndarray) -> set[tuple[int, int]]:
    tiers = np.array([a.tier for a in airports])
    pairs: set[tuple[int, int]] = set()

    def link(i: int, j: int) -> None:
        if i != j:
            pairs.add((min(i, j), max(i, j)))

    def nearest(i: int, mask: np.ndarray, k: int) -> list[int]:
        cand = np.flatnonzero(mask & (np.arange(len(airports)) != i))
        return [int(j) for j in cand[np.argsort(dist[i, cand], kind="stable")[:k]]]

    hubs = np.flatnonzero(tiers == 1)
    for i in hubs:
        for j in hubs:
            link(int(i), int(j))
    for i in np.flatnonzero(tiers == 2):
        for j in nearest(int(i), tiers == 1, 3) + nearest(int(i), tiers == 2, 3):
            link(int(i), j)
    for i in np.flatnonzero(tiers == 3):
        for j in nearest(int(i), tiers <= 2, 3):
            link(int(i), j)
    return pairs


def build_routes_graph(airports=None, edges=None, origin: str = "external") -> RoutesGraph:
    """Without `edges`, derive a hub-and-spoke graph from the seed tiers (tier-1 full mesh; tier-2 -> 3 nearest
    tier-1 + 3 nearest tier-2; tier-3 -> 3 nearest tier<=2). With `edges` (IATA pairs, e.g. from OpenFlights),
    use those and label the graph `origin`."""
    airports = tuple(airports) if airports is not None else load_airports()
    codes = tuple(a.iata for a in airports)
    index = {c: i for i, c in enumerate(codes)}
    dist = _distance_matrix(airports)
    if edges is None:
        pairs, origin = _derived_pairs(airports, dist), "derived"
    else:
        pairs = set()
        for a, b in edges:
            if a not in index or b not in index:
                raise ValueError(f"unknown airport in edge: {a}-{b}")
            if a != b:
                pairs.add((min(index[a], index[b]), max(index[a], index[b])))
    n = len(codes)
    if pairs:
        rows, cols = zip(*sorted(pairs))
        w = dist[list(rows), list(cols)]
        adj = sparse.coo_matrix((np.r_[w, w], (np.r_[rows, cols], np.r_[cols, rows])), shape=(n, n)).tocsr()
    else:
        adj = sparse.csr_matrix((n, n))
    return RoutesGraph(codes, index, adj, origin)


@lru_cache(maxsize=1)
def routes_graph() -> RoutesGraph:
    return build_routes_graph()
