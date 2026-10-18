"""Bird migration altitude band and strike-risk index. PRIOR-BASED ESTIMATE, not a measurement.

No free real radar or strike labels are used (rule 1), so this is an ecology prior:
  season   two migration peaks (spring ~25 Apr, autumn ~28 Sep), mirrored in the southern hemisphere, weak in the tropics;
  diel     nocturnal migration: night 1.0, twilight ~0.75, day 0.4 (local solar time);
  flyway   major corridors 1.0, others lower (a coarse longitude/latitude label);
  eBird    optional abundance blend, only when the eBird payload is real data (live or cached), never simulated.
Altitude bands are typical radar-ornithology ranges (night: ~500-5,000 ft AGL, up to ~8,000 ft on peak nights; day: 0-2,000 ft).
The ensemble uses the prior-only index so training and live inputs match; the eBird-blended index is for display.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

LABEL = "prior-based estimate"
_MAX_PRIOR = 0.8  # leaves headroom for the eBird blend
_BASE_SEASON = 0.45

_FLYWAYS = (  # (name, factor, predicate(lat, lon))
    ("Pacific Americas", 0.9, lambda la, lo: (la >= 15) & (lo >= -170) & (lo < -112)),
    ("Central Americas", 0.9, lambda la, lo: (la >= 15) & (lo >= -112) & (lo < -100)),
    ("Mississippi Americas", 1.0, lambda la, lo: (la >= 15) & (lo >= -100) & (lo < -82)),
    ("Atlantic Americas", 1.0, lambda la, lo: (la >= 15) & (lo >= -82) & (lo <= -50)),
    ("East Atlantic", 1.0, lambda la, lo: (la >= 30) & (lo >= -25) & (lo < 10)),
    ("Black Sea-Mediterranean", 1.0, lambda la, lo: (la >= 30) & (lo >= 10) & (lo < 45)),
    ("Central Asian", 0.9, lambda la, lo: (la >= 25) & (lo >= 45) & (lo < 90)),
    ("East Asian-Australasian", 1.0, lambda la, lo: (la >= -45) & (lo >= 90) & (lo <= 165)),
)


def _utc_parts(ts: np.ndarray):
    idx = pd.DatetimeIndex(pd.to_datetime(np.ravel(ts), unit="s"))
    return idx.dayofyear.to_numpy(), idx.hour.to_numpy() + idx.minute.to_numpy() / 60 + idx.second.to_numpy() / 3600


def solar_elevation(lat, lon, ts):
    """Solar elevation in degrees (NOAA general solar position formulas). Vectorised over lat/lon/ts."""
    lat, lon, ts = (np.asarray(a, dtype=float) for a in (lat, lon, ts))
    shape = np.broadcast(lat, lon, ts).shape
    doy, hour = _utc_parts(np.broadcast_to(ts, shape))
    g = 2 * np.pi / 365 * (doy - 1 + (hour - 12) / 24)
    eqtime = 229.18 * (0.000075 + 0.001868 * np.cos(g) - 0.032077 * np.sin(g) - 0.014615 * np.cos(2 * g) - 0.040849 * np.sin(2 * g))
    decl = 0.006918 - 0.399912 * np.cos(g) + 0.070257 * np.sin(g) - 0.006758 * np.cos(2 * g) + 0.000907 * np.sin(2 * g) - 0.002697 * np.cos(3 * g) + 0.00148 * np.sin(3 * g)
    tst = hour * 60 + eqtime + 4 * np.broadcast_to(lon, shape).ravel()
    ha = np.radians(tst / 4 - 180)
    phi = np.radians(np.broadcast_to(lat, shape).ravel())
    sin_e = np.sin(phi) * np.sin(decl) + np.cos(phi) * np.cos(decl) * np.cos(ha)
    return np.degrees(np.arcsin(np.clip(sin_e, -1, 1))).reshape(shape)[()]


def flyway(lat: float, lon: float) -> str:
    for name, _, pred in _FLYWAYS:
        if bool(pred(lat, lon)):
            return name
    return "unclassified"


def _flyway_factor(lat: np.ndarray, lon: np.ndarray) -> np.ndarray:
    out = np.full(np.shape(lat), 0.7)
    for _, factor, pred in reversed(_FLYWAYS):
        out = np.where(pred(lat, lon), factor, out)
    return out


def _gauss(x, centre, width):
    return np.exp(-0.5 * ((x - centre) / width) ** 2)


def _season(lat: np.ndarray, doy: np.ndarray) -> np.ndarray:
    eff = np.where(lat >= 0, doy, (doy + 181) % 365 + 1)
    amp = 0.15 + 0.85 * np.clip(np.abs(lat) / 35.0, 0, 1)  # weak seasonality in the tropics
    peak = np.maximum(_gauss(eff, 115, 25), 0.8 * _gauss(eff, 270, 32))
    return _BASE_SEASON + (1 - _BASE_SEASON) * amp * peak


def prior_index(lat, lon, ts):
    """Strike-risk prior in 0..0.8 (season x diel x flyway). Vectorised."""
    lat, lon, ts = (np.asarray(a, dtype=float) for a in (lat, lon, ts))
    shape = np.broadcast(lat, lon, ts).shape
    lat, lon, ts = (np.broadcast_to(a, shape) for a in (lat, lon, ts))
    doy, _ = _utc_parts(ts)
    elev = np.asarray(solar_elevation(lat, lon, ts)).ravel()
    diel = np.interp(elev, [-6, 0, 12], [1.0, 0.75, 0.4])
    idx = _season(lat.ravel(), doy) * diel * _flyway_factor(lat.ravel(), lon.ravel())
    return (_MAX_PRIOR * idx).reshape(shape)[()]


def flight_bird_index(o_lat, o_lon, t_dep, d_lat, d_lon, t_arr):
    """Ensemble input for a flight: mean prior at the departure and arrival airports at their times."""
    return (prior_index(o_lat, o_lon, t_dep) + prior_index(d_lat, d_lon, t_arr)) / 2


def _band(elev: float, season: float) -> tuple[int, int]:
    if elev < -6:  # night: peak migration, taller on peak nights
        return 500, int(5000 + 3000 * (season - _BASE_SEASON) / (1 - _BASE_SEASON))
    if elev < 6:
        return 0, 3500
    return 0, 2000


def bird_risk(lat: float, lon: float, ts: float, ebird: dict | None = None) -> dict:
    """Strike-risk index (0..1) and migration altitude band (ft AGL) at a place and time. `ebird` is a connector payload."""
    prior = float(prior_index(lat, lon, ts))
    elev = float(solar_elevation(lat, lon, ts))
    doy, _ = _utc_parts(np.asarray(ts))
    band = _band(elev, float(_season(np.asarray(lat), doy)[0]))
    index, source, abundance = prior, "prior", None
    if ebird and ebird.get("source") in {"live", "cached"} and ebird.get("individuals") is not None:
        abundance = float(np.clip(np.log1p(ebird["individuals"]) / np.log1p(1500), 0, 1))
        index, source = min(1.0, prior * (0.6 + 0.8 * abundance)), "prior+ebird"
    return {
        "risk_index": round(index, 4),
        "prior_index": round(prior, 4),
        "abundance_factor": abundance,
        "altitude_band_ft": band,
        "flyway": flyway(lat, lon),
        "solar_elevation_deg": round(elev, 1),
        "source": source,
        "label": LABEL,
    }
