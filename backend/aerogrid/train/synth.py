"""SYNTHETIC data generator for smoke tests and demos only. Never used for reported accuracy.

Produces the same Tables schema as train/real.py from a latent weather-severity process (AR(1) with a regional factor),
a load process and a cascade term, so the full pipeline can be exercised offline. Truth is built in on purpose:
weather, congestion, cascade, carrier and route effects exist; bird has NO effect (a fitted bird weight must be ~0).
Cause columns follow the BTS convention: zero unless the flight is 15+ minutes late, and they sum to the arrival delay.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from .. import data
from ..ml import bird, routing
from ..ml import weather as wx
from . import Tables

CARRIERS = ["AA", "DL", "UA", "WN", "B6", "AS"]
CARRIER_EFFECT = np.array([3.0, 2.0, 4.0, 1.5, 3.5, 2.5])
_SIZE = {1: 9.0, 2: 5.0, 3: 2.5}
_LOAD = {1: 0.95, 2: 0.8, 3: 0.6}
NOTE = "SYNTHETIC data from train/synth.py: pipeline smoke test only, not real flights, not for reported accuracy."


def _ar1(shape, phi, sd, rng):
    out = np.zeros(shape)
    eps = rng.normal(0, sd, shape)
    for t in range(1, shape[-1]):
        out[..., t] = phi * out[..., t - 1] + eps[..., t]
    return out


def _obs_from_sev(sev, rng):
    n = lambda s: rng.normal(0, s, sev.shape)  # noqa: E731
    vis = np.clip(10 - 6.5 * sev + n(0.6), 0.25, 10)
    ceil = np.clip(12 - 9 * sev + n(0.8), 0.2, 12)
    wind = np.clip(6 + 7 * sev + np.abs(n(2)), 0, 60)
    gust = wind + np.clip(3 + 9 * sev * rng.random(sev.shape) + np.abs(n(1)), 0, 40)
    cat = np.select([(vis < 1) | (ceil < 0.5), (vis < 3) | (ceil < 1.0), (vis <= 5) | (ceil <= 3.0)], [3.0, 2.0, 1.0], 0.0)
    return {
        "obs_vis_sm": vis, "obs_ceiling_kft": ceil, "obs_wind_kt": wind, "obs_gust_kt": gust,
        "obs_precip": np.where(sev > 0.5, np.where(sev > 1.2, 2.0, 1.0), 0.0),
        "obs_tstorm": ((sev > 1.0) & (rng.random(sev.shape) < 0.6)).astype(float), "obs_cat": cat,
    }


def _fc_from_sev(sev, rng):
    s = np.clip(sev + rng.normal(0, 0.25, sev.shape), 0, None)
    n = lambda sd: rng.normal(0, sd, sev.shape)  # noqa: E731
    return {
        "fc_precip_mm": 3 * np.maximum(0, s - 0.4) * (0.5 + rng.random(sev.shape)),
        "fc_gust_kt": np.clip(10 + 14 * s + np.abs(n(2)), 0, 80),
        "fc_cloud_low_pct": np.clip(25 + 55 * s + n(8), 0, 100),
        "fc_vis_km": np.clip(10 - 6 * s + n(0.6), 0.3, 10),
        "fc_cape_jkg": 1600 * np.maximum(0, s - 0.8) * rng.random(sev.shape),
    }


def make_synthetic(n_days: int = 45, n_airports: int | None = None, seed: int = 0, flights_scale: float = 0.5, start: str = "2025-03-03") -> Tables:
    rng = np.random.default_rng(seed)
    us = sorted((a for a in data.load_airports() if a.country in {"US", "PR"}), key=lambda a: (a.tier, a.iata))
    nodes = us[:n_airports] if n_airports else us
    n, hours = len(nodes), n_days * 24
    ts0 = pd.Timestamp(start, tz="UTC").timestamp()
    hour_ts = ts0 + 3600.0 * np.arange(hours)
    iata = [a.iata for a in nodes]
    lat, lon = np.array([a.lat for a in nodes]), np.array([a.lon for a in nodes])
    tier = np.array([a.tier for a in nodes])
    dist = data._distance_matrix(tuple(nodes))

    # latent severity: regional factor shared by airports in the same longitude band + local AR(1)
    band = np.digitize(lon, [-110, -95, -80])
    z = 0.5 * _ar1((4, hours), 0.93, 0.35, rng)[band] + 0.85 * _ar1((n, hours), 0.93, 0.35, rng)
    sev = 1.2 * np.maximum(0.0, z - 0.5)
    horizon = rng.choice([2, 6, 24], hours)  # one issue-time horizon per snapshot
    issue = np.clip(np.arange(hours) - horizon, 0, None)
    obs, fc = _obs_from_sev(sev, rng), _fc_from_sev(sev, rng)

    # load and cascade
    local_hour = ((hour_ts[None, :] + lon[:, None] * 240.0) % 86400) // 3600
    diurnal = 0.15 + 0.85 * np.exp(-0.5 * ((local_hour - 13) / 5) ** 2)
    load = diurnal * np.array([_LOAD[t] for t in tier])[:, None]
    size = np.array([_SIZE[t] for t in tier])
    weights = np.outer(size, size) / (1 + dist / 1000.0)
    np.fill_diagonal(weights, 0.0)
    a_in = weights / weights.sum(axis=1, keepdims=True)
    sev_lag = np.concatenate([np.zeros((n, 2)), sev[:, :-2]], axis=1)
    cascade = a_in @ sev_lag
    counts = rng.poisson(size[:, None] * flights_scale * diurnal)

    # flights
    ai, ti = np.nonzero(counts)
    rep = counts[ai, ti]
    o, t_idx = np.repeat(ai, rep), np.repeat(ti, rep)
    f = len(o)
    d = np.empty(f, dtype=int)
    for i in range(n):
        m = o == i
        if m.any():
            d[m] = rng.choice(n, size=int(m.sum()), p=weights[i] / weights[i].sum())
    carrier = rng.choice(len(CARRIERS), f)
    ts_hour = hour_ts[t_idx]
    ts_dep = ts_hour + rng.integers(0, 3600, f)
    distance = dist[o, d]
    crs = distance / 13.27 + 28.0
    ts_arr = ts_dep + crs * 60
    arr_idx = np.clip(((ts_arr - ts0) // 3600).astype(int), 0, hours - 1)
    iss = issue[t_idx]
    frac = {k: np.clip(((ts_dep + crs * 60 * k - ts0) // 3600).astype(int), 0, hours - 1) for k in (0.25, 0.5, 0.75)}
    sev_er = np.stack([(1 - k) * sev[o, frac[k]] + k * sev[d, frac[k]] + np.abs(rng.normal(0, 0.15, f)) for k in (0.25, 0.5, 0.75)])
    er_fc = _fc_from_sev(sev_er, rng)
    cols = {
        "origin": np.array(iata)[o], "dest": np.array(iata)[d], "carrier": np.array(CARRIERS)[carrier],
        "ts_dep": ts_dep, "ts_arr": ts_arr, "ts_hour": ts_hour, "crs_elapsed_min": crs, "distance_km": distance, "horizon_h": horizon[t_idx],
        "o_lon": lon[o], "o_tier": tier[o], "d_tier": tier[d],
        "er_max_gust_kt": er_fc["fc_gust_kt"].max(axis=0), "er_max_precip_mm": er_fc["fc_precip_mm"].max(axis=0),
        "er_max_cape_jkg": er_fc["fc_cape_jkg"].max(axis=0), "er_max_cloud_low_pct": er_fc["fc_cloud_low_pct"].max(axis=0),
    }
    months = pd.DatetimeIndex(pd.to_datetime(ts_dep, unit="s")).month.to_numpy()
    zone_of = {(i, m): data.weather_zone(float(lat[i]), float(lon[i]), int(m)) for i in range(n) for m in np.unique(months)}
    cols["o_zone"] = np.array([zone_of[(i, m)] for i, m in zip(o, months)])
    for k, v in obs.items():
        cols[f"o_{k}"], cols[f"d_{k}"] = v[o, iss], v[d, iss]
    for k, v in fc.items():
        cols[f"o_{k}"], cols[f"d_{k}"] = v[o, t_idx], v[d, arr_idx]

    # route and bird components (the same functions the live app uses)
    jet_speed, jet_from = np.clip(rng.normal(70, 25, (3, f)), 0, None), rng.normal(270, 30, (3, f))
    cols["route_extra_min"] = routing.wind_extra_minutes_vec(lat[o], lon[o], lat[d], lon[d], jet_speed, jet_from)["extra_min"]
    cols["bird_index"] = bird.flight_bird_index(lat[o], lon[o], ts_dep, lat[d], lon[d], ts_arr)

    # causes (truth): weather, NAS (congestion + en-route), late aircraft (cascade), carrier, security
    exp = lambda: rng.exponential(1.0, f)  # noqa: E731
    sev_flight = np.maximum.reduce([sev[o, t_idx], sev[d, arr_idx], sev_er.max(axis=0)])
    weather = 11 * sev_flight * exp()
    l_o = load[o, t_idx]
    nas = (9 * sev[o, t_idx] * l_o + 5 * np.maximum(0, l_o - 0.85)) * exp() + 0.5 * cols["route_extra_min"]
    late = 12 * cascade[o, t_idx] * exp()
    carr = CARRIER_EFFECT[carrier] * exp() + exp()
    sec = (rng.random(f) < 0.004) * rng.exponential(25.0, f)
    total = weather + nas + late + carr + sec
    is_late = total >= 15
    parts = {"weather_delay": weather, "nas_delay": nas, "late_aircraft_delay": late, "carrier_delay": carr, "security_delay": sec}
    for k, v in parts.items():
        cols[k] = np.where(is_late, np.round(v, 1), 0.0)
    cols["arr_delay"] = np.where(is_late, np.round(sum(cols[k] for k in parts)), np.floor(total * 0.6)).astype(float)
    cols["dep_delay"] = np.floor(0.85 * cols["arr_delay"])
    cols["flight_id"] = np.arange(f)
    flights = pd.DataFrame(cols)

    # airport-hour table
    grid_a, grid_t = np.meshgrid(np.arange(n), np.arange(hours), indexing="ij")
    ah = pd.DataFrame({
        "airport": np.array(iata)[grid_a.ravel()], "ts": hour_ts[grid_t.ravel()], "horizon_h": horizon[grid_t.ravel()],
        "tier": tier[grid_a.ravel()], "lon": lon[grid_a.ravel()],
        **{k: v[grid_a.ravel(), issue[grid_t.ravel()]] for k, v in obs.items()},
        **{k: v[grid_a.ravel(), grid_t.ravel()] for k, v in fc.items()},
    })
    amonth = pd.DatetimeIndex(pd.to_datetime(ah["ts"], unit="s")).month.to_numpy()
    zone_of = {(i, m): data.weather_zone(float(lat[i]), float(lon[i]), int(m)) for i in range(n) for m in np.unique(amonth)}
    ah["zone"] = [zone_of[(i, m)] for i, m in zip(grid_a.ravel(), amonth)]
    agg = flights.groupby(["origin", "ts_hour"]).agg(n=("flight_id", "size"), y_dep=("dep_delay", "mean"), y_late=("late_aircraft_delay", "mean"), y_nas=("nas_delay", "mean"))
    agg.index.names = ["airport", "ts"]
    ah = ah.join(agg, on=["airport", "ts"])
    ah["n"] = ah["n"].fillna(0).astype(int)

    day = 86400
    train_end = ts0 + int(0.7 * n_days) * day
    calib_start = train_end + day
    calib_end = calib_start + max(1, round(0.1 * n_days)) * day
    split = {"train_end": train_end, "calib_start": calib_start, "calib_end": calib_end, "test_start": calib_end + day}
    airports = pd.DataFrame({"iata": iata, "tier": tier, "lat": lat, "lon": lon})
    meta = {"trained_on": "synthetic", "note": NOTE, "n_days": n_days, "seed": seed, "start": start, "n_flights": int(f)}
    return Tables(flights, ah, airports, iata, split, meta)
