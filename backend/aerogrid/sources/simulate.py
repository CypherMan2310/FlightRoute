"""Deterministic simulated fallbacks for every connector.

Everything returned here is labelled source="simulated" and must never be presented as real. Output is a pure
function of (inputs, time bucket): same call, same answer. Shapes match the normalised live payloads; helpers
that shape live records (flight_category, nas.summarise) are imported lazily to avoid a circular import.
"""
from __future__ import annotations

import math
import random
import zlib
from datetime import datetime, timezone

from .. import data
from .cache import NoSimulation

_KT_TO_KMH = 1.852
_US_CODES = {"US", "PR", "GU"}


def _rng(*parts) -> random.Random:
    return random.Random(zlib.crc32("|".join(str(p) for p in parts).encode()))


def _utc(t: float) -> datetime:
    return datetime.fromtimestamp(t, tz=timezone.utc)


# ---- weather -------------------------------------------------------------------------------------------------

def _temp_c(lat: float, lon: float, t: float) -> float:
    """Crude climatology: latitude mean, hemisphere-aware seasonal swing (peak mid-July in the north), diurnal swing."""
    dt = _utc(t)
    doy = dt.timetuple().tm_yday
    local_hour = (dt.hour + dt.minute / 60 + lon / 15) % 24
    mean = 27 - 0.0075 * lat * lat
    seasonal = 0.25 * abs(lat) * math.cos(2 * math.pi * (doy - 200) / 365) * (1 if lat >= 0 else -1)
    diurnal = 4 * math.cos(2 * math.pi * (local_hour - 15) / 24)
    return mean + seasonal + diurnal


def _weather(lat: float, lon: float, hour_bucket: int, key: str) -> dict:
    t = hour_bucket * 3600
    month = _utc(t).month
    zone = data.weather_zone(lat, lon, month)
    rng = _rng("wx", key, hour_bucket)
    temp = round(_temp_c(lat, lon, t) + rng.gauss(0, 1.5), 1)
    p_rain = {"tropical": 0.25, "monsoon": 0.45, "subtropical": 0.12, "temperate": 0.2, "subpolar": 0.25}[zone]
    rain = rng.random() < p_rain
    storm = rain and zone in {"tropical", "monsoon", "subtropical"} and rng.random() < 0.2
    precip = round(rng.uniform(0.2, 6.0), 1) if rain else 0.0
    spread = rng.uniform(0.5, 3.0) if zone in {"tropical", "monsoon"} or rain else rng.uniform(1.0, 9.0)
    fog = (not rain) and zone in {"temperate", "subpolar"} and spread < 2.0 and rng.random() < 0.3
    cover_pct = rng.uniform(80, 100) if rain else rng.choice((0, 5, 15, 25, 40, 60, 75, 95))
    if cover_pct < 10:
        layers: list[dict] = []
    else:
        cover = "FEW" if cover_pct < 30 else "SCT" if cover_pct < 55 else "BKN" if cover_pct < 90 else "OVC"
        base = rng.randrange(600, 2500, 100) if rain or fog else rng.randrange(1500, 12000, 100)
        layers = [{"cover": cover, "base_ft": base}]
    wind = min(35, int(rng.gammavariate(2.0, 5.0)))
    gust = wind + rng.randint(6, 14) if (wind >= 12 and rng.random() < 0.4) or storm else None
    if storm:
        wx, code, vis = "TSRA", 95, round(rng.uniform(1.0, 4.0), 1)
    elif rain:
        wx = "-RA" if precip < 1 else "RA" if precip < 3 else "+RA"
        code, vis = (61 if precip < 1 else 63 if precip < 3 else 65), round(rng.uniform(2.0, 7.0), 1)
    elif fog:
        wx, code, vis = "FG", 45, round(rng.uniform(0.2, 0.9), 1)
    else:
        wx, code, vis = "", (0 if cover_pct < 10 else 1 if cover_pct < 40 else 2 if cover_pct < 75 else 3), 10.0
    jet = (30 + 70 * math.sin(min(abs(lat), 60) / 60 * math.pi)) * (1.3 if (month in (11, 12, 1, 2, 3)) == (lat >= 0) else 0.8)
    westerly = abs(lat) >= 25
    return {
        "cloud_low_pct": int(cover_pct * rng.uniform(0.2, 1.0)),
        "cape_jkg": round(rng.uniform(800, 3000), 0) if storm else (round(rng.uniform(0, 300), 0) if rain else 0.0),
        "wind250_kt": round(max(0.0, rng.gauss(jet, 15)), 1),
        "wind250_dir_deg": int((270 if westerly else 90) + rng.gauss(0, 25)) % 360,
        "time": t,
        "temp_c": temp,
        "dewpoint_c": round(temp - spread, 1),
        "wind_dir_deg": rng.randrange(0, 360, 10),
        "wind_kt": wind,
        "gust_kt": gust,
        "visibility_sm": vis,
        "altimeter_hpa": round(rng.gauss(1013, 7), 1),
        "wx": wx,
        "clouds": layers,
        "ceiling_ft": next((c["base_ft"] for c in layers if c["cover"] in {"BKN", "OVC"}), None),
        "precip_mm": precip,
        "cloud_cover_pct": int(cover_pct),
        "weather_code": code,
    }


def sim_metar(icao: str, now: float) -> dict:
    from .metar import flight_category

    try:
        a = data.get_airport(icao)
    except KeyError:
        raise NoSimulation(icao) from None
    w = _weather(a.lat, a.lon, int(now // 3600), a.icao)
    return {
        "icao": a.icao,
        "name": a.name,
        "lat": a.lat,
        "lon": a.lon,
        "obs_time": w["time"],
        "temp_c": w["temp_c"],
        "dewpoint_c": w["dewpoint_c"],
        "wind_dir_deg": w["wind_dir_deg"],
        "wind_kt": w["wind_kt"],
        "gust_kt": w["gust_kt"],
        "visibility_sm": w["visibility_sm"],
        "altimeter_hpa": w["altimeter_hpa"],
        "wx": w["wx"],
        "clouds": w["clouds"],
        "ceiling_ft": w["ceiling_ft"],
        "flight_category": flight_category(w["visibility_sm"], w["ceiling_ft"]),
        "raw": None,
        "source": "simulated",
    }


def sim_taf(icao: str, now: float) -> dict:
    """Three 6-hour periods built from simulated METAR states (a persistence-free toy forecast)."""
    start = int(now // 3600) * 3600
    periods = []
    for k in range(3):
        m = sim_metar(icao, start + k * 6 * 3600)
        periods.append(
            {
                "from": start + k * 6 * 3600,
                "to": start + (k + 1) * 6 * 3600,
                "change": None if k == 0 else "FM",
                "probability": None,
                **{f: m[f] for f in ("wind_dir_deg", "wind_kt", "gust_kt", "visibility_sm", "wx", "clouds", "ceiling_ft", "flight_category")},
            }
        )
    return {"icao": icao, "issue_time": start, "valid_from": start, "valid_to": start + 18 * 3600, "raw": None, "periods": periods, "source": "simulated"}


def sim_forecast(lat: float, lon: float, now: float, hours: int) -> dict:
    start = int(now // 3600)
    hourly = []
    for h in range(hours):
        w = _weather(lat, lon, start + h, f"{lat:.2f},{lon:.2f}")
        hourly.append(
            {
                "time": w["time"],
                "temp_c": w["temp_c"],
                "precip_mm": w["precip_mm"],
                "wind_kt": float(w["wind_kt"]),
                "gust_kt": float(w["gust_kt"] if w["gust_kt"] is not None else w["wind_kt"]),
                "visibility_m": round(w["visibility_sm"] * 1609.344, 0),
                "cloud_low_pct": w["cloud_low_pct"],
                "cape_jkg": w["cape_jkg"],
                "weather_code": w["weather_code"],
                "wind250_kt": w["wind250_kt"],
                "wind250_dir_deg": w["wind250_dir_deg"],
            }
        )
    return {"lat": lat, "lon": lon, "hourly": hourly, "source": "simulated"}


# ---- traffic -------------------------------------------------------------------------------------------------

def sim_flights(lat: float, lon: float, radius_nm: float, now: float) -> dict:
    """Synthetic aircraft flying the *derived* route graph (callsigns SIMxxxx, icao24 f0xxxx). Not real traffic."""
    from scipy import sparse

    graph = data.routes_graph()
    tri = sparse.triu(graph.adjacency).tocoo()
    flights = []
    idx = 0
    for i, j, dist_km in zip(tri.row, tri.col, tri.data):
        a, b = data.get_airport(graph.codes[i]), data.get_airport(graph.codes[j])
        for k in range(_rng("edge", a.iata, b.iata).choice((0, 1, 1, 2))):
            idx += 1
            rng = _rng("flight", a.iata, b.iata, k)
            speed_kt = rng.uniform(430, 480)
            airborne_s = dist_km / (speed_kt * _KT_TO_KMH) * 3600
            cycle = airborne_s * 1.5
            frac = (now / cycle + rng.random()) % 1.0
            if frac >= 2 / 3:
                continue  # on the ground between legs: not listed
            f = frac * 1.5
            src, dst = (a, b) if k % 2 == 0 else (b, a)
            flat, flon = data.great_circle_point(src.lat, src.lon, dst.lat, dst.lon, f)
            if data.great_circle_km(lat, lon, flat, flon) > radius_nm * _KT_TO_KMH:
                continue
            nlat, nlon = data.great_circle_point(src.lat, src.lon, dst.lat, dst.lon, min(1.0, f + 0.01))
            track = data.initial_bearing(flat, flon, nlat, nlon)
            edge = min(f, 1 - f)
            flights.append(
                {
                    "icao24": f"{0xF00000 + idx:06x}",
                    "callsign": f"SIM{idx % 10000:04d}",
                    "lat": round(flat, 5),
                    "lon": round(flon, 5),
                    "alt_ft": round(max(1500.0, 36000 * min(1.0, edge / 0.12)), -2),
                    "on_ground": False,
                    "gs_kt": round(speed_kt * (0.6 + 0.4 * min(1.0, edge / 0.12)), 1),
                    "track_deg": round(track, 1),
                    "vrate_fpm": 1800 if f < 0.12 else -1500 if f > 0.88 else 0,
                    "aircraft_type": None,
                }
            )
    return {"provider": "simulated", "center": {"lat": lat, "lon": lon}, "radius_nm": radius_nm, "count": len(flights), "flights": flights, "source": "simulated"}


# ---- FAA NAS status ------------------------------------------------------------------------------------------

_NAS_REASONS = ("WX:Thunderstorms", "WX:Low Ceilings", "WX:Wind", "VOL:Volume", "RWY:Maintenance")


def sim_nas(now: float) -> dict:
    from .nas import summarise

    bucket = int(now // 1800)
    rng = _rng("nas", bucket)
    pool = sorted((a for a in data.load_airports() if a.country in _US_CODES), key=lambda a: a.iata)
    weights = {1: 4, 2: 2, 3: 1}
    chosen = []
    for _ in range(rng.choices((0, 1, 2, 3, 5), (25, 30, 20, 15, 10))[0]):
        a = rng.choices(pool, [weights[p.tier] for p in pool])[0]
        pool.remove(a)
        chosen.append(a)
    airports = {}
    for a in chosen:
        kind = rng.choices(("general_delay", "ground_delay", "ground_stop"), (70, 20, 10))[0]
        reason = rng.choice(_NAS_REASONS)
        if kind == "general_delay":
            m = rng.choice((15, 30, 45, 60, 90))
            prog = {"type": kind, "direction": rng.choice(("arrival", "departure")), "reason": reason, "min_minutes": max(0, m - 14), "max_minutes": m, "trend": "Steady"}
        elif kind == "ground_delay":
            m = rng.choice((45, 90, 120, 180))
            prog = {"type": kind, "reason": reason, "avg_minutes": int(m * 0.7), "max_minutes": m}
        else:
            prog = {"type": kind, "reason": reason, "end_time": None}
        airports[a.iata] = summarise([prog])
    return {"provider": "simulated", "updated": bucket * 1800, "airports": airports, "notams": [], "source": "simulated"}


# ---- eBird ---------------------------------------------------------------------------------------------------

def sim_ebird(lat: float, lng: float, now: float, dist_km: float = 25, back_days: int = 14) -> dict:
    """Seasonal toy abundance: spring (Apr-May) and autumn (Sep-Oct) migration peaks, mirrored in the south,
    flatter in the tropics. Generic species names on purpose."""
    dt = _utc(now)
    x = dt.month + (dt.day - 1) / 30 + (0 if lat >= 0 else -6)
    x = (x - 1) % 12 + 1

    def peak(c: float) -> float:
        d = min(abs(x - c), 12 - abs(x - c))
        return math.exp(-((d / 1.2) ** 2))

    amp = 1.6 * min(1.0, abs(lat) / 30) + 0.2
    factor = 0.4 + amp * (peak(4.5) + peak(9.75))
    rng = _rng("ebird", round(lat, 1), round(lng, 1), dt.month)
    individuals = int(600 * factor * rng.uniform(0.9, 1.1))
    n_records = individuals // 12
    n_species = min(n_records, int(8 + 25 * factor))
    shares = [1 / (r + 1) for r in range(min(5, n_species))]
    top = [{"code": f"sim{r + 1}", "name": f"simulated species {r + 1}", "individuals": int(individuals * s / sum(shares) * 0.8)} for r, s in enumerate(shares)]
    return {
        "provider": "simulated",
        "lat": lat,
        "lng": lng,
        "dist_km": dist_km,
        "back_days": back_days,
        "n_records": n_records,
        "n_species": n_species,
        "individuals": individuals,
        "top_species": top,
        "source": "simulated",
    }
