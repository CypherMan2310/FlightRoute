"""What-if: drop a storm on an airport or close part of its runway capacity and watch the cascade through the graph.

A MODEL-BASED SCENARIO, not a forecast: the scenario edits that airport's model inputs, the congestion model and the GCN are re-run
on the world graph, and the difference to the baseline run is reported per airport with its distance (hops) from the scenario airport.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from scipy.sparse.csgraph import shortest_path

from .. import data

LABEL = "model-based scenario (not a forecast)"
KINDS = ("storm", "runway_closure")
MAX_CAPACITY_LOSS = 0.6  # severity 1 removes 60% of runway capacity (roughly 2 of 3 runways); scheduled demand is scaled by 1 / (1 - loss)
ASSUMPTIONS = {
    "storm": "Thunderstorm over the airport: visibility, ceiling, wind, gusts, precipitation, CAPE and low cloud move toward severe values in proportion to severity.",
    "runway_closure": f"Runway capacity loss of up to {int(MAX_CAPACITY_LOSS * 100)}%: typical scheduled departures and arrivals are scaled by 1 / (1 - loss) to mimic demand against reduced capacity.",
}


def _lerp(a, b, s):
    return a + (b - a) * s


def apply_scenario(raw: pd.DataFrame, airport: str, kind: str, severity: float) -> pd.DataFrame:
    if kind not in KINDS:
        raise ValueError(f"kind must be one of {KINDS}")
    if not 0.0 <= severity <= 1.0:
        raise ValueError("severity must be between 0 and 1")
    iata = data.get_airport(airport).iata
    out = raw.copy()
    i = out.index[out["airport"] == iata][0]
    s = float(severity)
    if kind == "runway_closure":
        scale = 1.0 / (1.0 - MAX_CAPACITY_LOSS * s)
        out.loc[i, ["sched_deps", "sched_arrs"]] = out.loc[i, ["sched_deps", "sched_arrs"]].to_numpy(dtype=float) * scale
        return out
    cur = out.loc[i]
    new = {
        "obs_vis_sm": _lerp(cur["obs_vis_sm"], min(cur["obs_vis_sm"], 0.5), s), "obs_ceiling_kft": _lerp(cur["obs_ceiling_kft"], min(cur["obs_ceiling_kft"], 0.8), s),
        "obs_wind_kt": _lerp(cur["obs_wind_kt"], max(cur["obs_wind_kt"], 28.0), s), "obs_gust_kt": _lerp(cur["obs_gust_kt"], max(cur["obs_gust_kt"], 45.0), s),
        "obs_precip": cur["obs_precip"] if s == 0 else max(cur["obs_precip"], 2.0 if s >= 0.66 else 1.0),
        "obs_tstorm": 1.0 if s >= 0.5 else cur["obs_tstorm"],
        "obs_cat": max(cur["obs_cat"], 3.0 if s >= 0.8 else 2.0) if s >= 0.4 else cur["obs_cat"],
        "fc_precip_mm": _lerp(cur["fc_precip_mm"], max(cur["fc_precip_mm"], 8.0), s), "fc_gust_kt": _lerp(cur["fc_gust_kt"], max(cur["fc_gust_kt"], 45.0), s),
        "fc_cloud_low_pct": _lerp(cur["fc_cloud_low_pct"], 100.0, s), "fc_vis_km": _lerp(cur["fc_vis_km"], min(cur["fc_vis_km"], 1.0), s),
        "fc_cape_jkg": _lerp(cur["fc_cape_jkg"], max(cur["fc_cape_jkg"], 2500.0), s),
    }
    for k, v in new.items():
        out.loc[i, k] = float(v)
    return out


def run(bundle, raw: pd.DataFrame, airport: str, kind: str, severity: float, top: int = 15) -> dict:
    after_raw = apply_scenario(raw, airport, kind, severity)  # validates the inputs
    iata = data.get_airport(airport).iata
    before, after = bundle.score_world(raw), bundle.score_world(after_raw)
    g = data.routes_graph()
    hops = shortest_path(g.adjacency, unweighted=True, directed=False, indices=g.index[iata])
    deltas = []
    for j, a in enumerate(data.load_airports()):
        deltas.append({
            "iata": a.iata, "name": a.name, "lat": a.lat, "lon": a.lon, "hops": int(hops[j]) if np.isfinite(hops[j]) else None,
            "dep_before": float(before["gcn"][j, 0]), "dep_after": float(after["gcn"][j, 0]),
            "delta_dep_min": float(after["gcn"][j, 0] - before["gcn"][j, 0]), "delta_late_min": float(after["gcn"][j, 1] - before["gcn"][j, 1]),
            "stress_before": float(before["stress"][j]), "stress_after": float(after["stress"][j]), "delta_stress": float(after["stress"][j] - before["stress"][j]),
        })
    affected = sorted(deltas, key=lambda d: -abs(d["delta_dep_min"]))[:top]
    return {"airport": iata, "kind": kind, "severity": float(severity), "deltas": deltas, "affected": affected, "label": LABEL, "trained_on": bundle.trained_on,
            "assumption": ASSUMPTIONS[kind], "inference_graph": "derived routes graph (not real schedules) until OpenFlights/BTS edges are loaded"}
