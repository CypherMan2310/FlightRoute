"""Top drivers inside each component.

* LightGBM components (weather, congestion): `pred_contrib` (exact, additive, in minutes).
* GCN (cascade): ablation. Each feature is set to its training mean at every node; the contribution is the drop in the node's
  prediction (it includes the effect through neighbouring airports). Exact for the model, not a causal claim.
"""
from __future__ import annotations

import numpy as np

_OBS = {"obs_vis_sm": "visibility (SM)", "obs_ceiling_kft": "cloud ceiling (kft)", "obs_wind_kt": "wind speed (kt)", "obs_gust_kt": "wind gusts (kt)",
        "obs_precip": "precipitation reported", "obs_tstorm": "thunderstorm reported", "obs_cat": "flight category (VFR to LIFR)"}
_FC = {"fc_precip_mm": "forecast precipitation (mm)", "fc_gust_kt": "forecast gusts (kt)", "fc_cloud_low_pct": "forecast low cloud (%)",
       "fc_vis_km": "forecast visibility (km)", "fc_cape_jkg": "forecast convective energy (CAPE)"}
_ER = {"er_max_gust_kt": "en-route strongest forecast gusts (kt)", "er_max_precip_mm": "en-route heaviest forecast precipitation (mm)",
       "er_max_cape_jkg": "en-route peak convective energy (CAPE)", "er_max_cloud_low_pct": "en-route most low cloud (%)"}
_MISC = {"distance_km": "route distance (km)", "crs_elapsed_min": "scheduled block time (min)", "tier_o": "origin hub tier", "tier_d": "destination hub tier",
         "tier": "airport hub tier", "holiday": "US holiday", "horizon_h": "forecast horizon (h)", "sched_deps": "typical scheduled departures this hour",
         "sched_arrs": "typical scheduled arrivals this hour", "hour_sin": "time of day", "hour_cos": "time of day", "dow_sin": "day of week", "dow_cos": "day of week",
         "month_sin": "season (month)", "month_cos": "season (month)",
         "hp_nas_ap": "airport's usual NAS-type delay", "hp_nas_how": "airport's usual NAS-type delay at this hour of the week",
         "hp_dep_ap": "airport's usual departure delay", "hp_dep_how": "airport's usual departure delay at this hour of the week",
         "hp_late_ap": "airport's usual late-aircraft delay", "hp_late_how": "airport's usual late-aircraft delay at this hour of the week"}


def label(feature: str) -> str:
    """Readable name for a model feature, e.g. 'o_obs_tstorm' -> 'thunderstorm reported at origin'."""
    if feature.startswith("zone_"):
        return f"weather zone: {feature[5:]}"
    for prefix, where in (("o_", "origin"), ("d_", "destination")):
        if feature.startswith(prefix) and feature[2:] in {**_OBS, **_FC}:
            return f"{({**_OBS, **_FC})[feature[2:]]} at {where}"
    for table in (_OBS, _FC, _ER, _MISC):
        if feature in table:
            return table[feature] + (" at the airport" if table is _OBS or table is _FC else "")
    return feature


def _top(features, values, contrib, k: int, merge: bool, **extra) -> list[dict]:
    """Largest |contribution| drivers; with merge, features sharing a readable label (sin/cos pairs) are summed into one driver."""
    groups: dict[str, dict] = {}
    for name, v, c in zip(features, values, contrib):
        key = label(name) if merge else name
        g = groups.setdefault(key, {"features": [], "value": float(v), "contribution_min": 0.0, "label": label(name)})
        g["features"].append(name)
        g["contribution_min"] += float(c)
    ranked = sorted(groups.values(), key=lambda g: -abs(g["contribution_min"]))[:k]
    return [{"feature": "+".join(g["features"]), "label": g["label"], "value": g["value"] if len(g["features"]) == 1 else None, "contribution_min": g["contribution_min"], **extra} for g in ranked]


def lgbm_drivers(model, X, k: int = 3, merge: bool = True) -> list[list[dict]]:
    """Per row: the k drivers with the largest |contribution| (minutes), with value and readable label."""
    contrib = model.contrib(X)[:, :-1]
    values = X[model.feature_names].to_numpy(dtype=float)
    return [_top(model.feature_names, v, c, k, merge) for c, v in zip(contrib, values)]


def gcn_drivers(model, ahat, x: np.ndarray, node: int, head: int, feature_names, k: int = 3, merge: bool = True) -> list[dict]:
    """x: raw (N, F) features of one snapshot. Contribution of feature j = prediction - prediction with feature j at its training mean (all nodes)."""
    n, f = x.shape
    variants = np.repeat(x[None], f, axis=0)
    for j in range(f):
        variants[j, :, j] = model.scaler.mean[j]
    base = float(model.predict(ahat, x)[node, head])
    contrib = base - model.predict(ahat, variants)[:, node, head]
    return _top(feature_names, x[node], contrib, k, merge, method="ablation to the training mean")
