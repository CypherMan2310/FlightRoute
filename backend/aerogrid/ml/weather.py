"""Weather-delay model (LightGBM) and the feature layer shared by every learned component.

Canonical weather inputs are identical at training and in the live app (train/serve parity):
  obs_*  from a METAR-like record (IEM ASOS in training, aviationweather.gov METAR live), taken at the issue time;
  fc_*   from an hourly forecast row (Open-Meteo historical-forecast in training, Open-Meteo forecast live), for the target hour.
Both raw frames go through the single vectorised `build_flight_features`. Local time is approximated by solar time (UTC + lon/15).
"""
from __future__ import annotations

import json
from functools import lru_cache
from pathlib import Path

import lightgbm as lgb
import numpy as np
import pandas as pd
from pandas.tseries.holiday import Holiday, USFederalHolidayCalendar

from .. import data

OBS = ["obs_vis_sm", "obs_ceiling_kft", "obs_wind_kt", "obs_gust_kt", "obs_precip", "obs_tstorm", "obs_cat"]
FC = ["fc_precip_mm", "fc_gust_kt", "fc_cloud_low_pct", "fc_vis_km", "fc_cape_jkg"]
CAL = ["hour_sin", "hour_cos", "dow_sin", "dow_cos", "month_sin", "month_cos", "holiday", "horizon_h"]
ZONE_COLS = [f"zone_{z}" for z in data.ZONES]
ENROUTE = ["er_max_gust_kt", "er_max_precip_mm", "er_max_cape_jkg", "er_max_cloud_low_pct"]
FLIGHT_FEATURES = [f"o_{c}" for c in OBS + FC] + [f"d_{c}" for c in OBS + FC] + ENROUTE + ["distance_km", "crs_elapsed_min", "tier_o", "tier_d"] + CAL + ZONE_COLS

NEUTRAL_OBS = {"obs_vis_sm": 10.0, "obs_ceiling_kft": 12.0, "obs_wind_kt": 5.0, "obs_gust_kt": 5.0, "obs_precip": 0.0, "obs_tstorm": 0.0, "obs_cat": 0.0}
NEUTRAL_FC = {"fc_precip_mm": 0.0, "fc_gust_kt": 8.0, "fc_cloud_low_pct": 0.0, "fc_vis_km": 10.0, "fc_cape_jkg": 0.0}
_CAT = {"VFR": 0.0, "MVFR": 1.0, "IFR": 2.0, "LIFR": 3.0}
_PRECIP = ("RA", "SN", "DZ", "GR", "GS", "PL", "SG", "IC", "UP")


def obs_features(rec: dict | None) -> dict:
    """Canonical observation features from a normalised METAR record (None -> neutral good weather)."""
    if not rec:
        return dict(NEUTRAL_OBS)
    w = rec.get("wx") or ""
    precip = 2.0 if "+" in w and any(p in w for p in _PRECIP) else 1.0 if any(p in w for p in _PRECIP) else 0.0
    vis = rec.get("visibility_sm")
    ceil = rec.get("ceiling_ft")
    wind = rec.get("wind_kt") or 0
    gust = rec.get("gust_kt")
    return {
        "obs_vis_sm": 10.0 if vis is None else float(min(vis, 10.0)),
        "obs_ceiling_kft": 12.0 if ceil is None else float(min(ceil / 1000.0, 12.0)),
        "obs_wind_kt": float(wind),
        "obs_gust_kt": float(wind if gust is None else gust),
        "obs_precip": precip,
        "obs_tstorm": 1.0 if "TS" in w else 0.0,
        "obs_cat": _CAT.get(rec.get("flight_category"), 0.0),
    }


def fc_features(h: dict | None) -> dict:
    """Canonical forecast features from one hourly forecast row (None -> neutral)."""
    if not h:
        return dict(NEUTRAL_FC)
    return {
        "fc_precip_mm": float(h["precip_mm"]),
        "fc_gust_kt": float(h["gust_kt"]),
        "fc_cloud_low_pct": float(h["cloud_low_pct"]),
        "fc_vis_km": float(min(h["visibility_m"] / 1000.0, 10.0)),
        "fc_cape_jkg": float(h["cape_jkg"]),
    }


@lru_cache(maxsize=1)
def _holiday_days() -> pd.DatetimeIndex:
    """US federal holidays 2005-2045: the observed day (pandas) plus the actual calendar date of fixed-date holidays."""
    cal = USFederalHolidayCalendar()
    start, end = pd.Timestamp("2005-01-01"), pd.Timestamp("2045-12-31")
    out = cal.holidays(start=start, end=end)
    for rule in cal.rules:
        if rule.month and rule.day and rule.observance is not None:
            out = out.union(Holiday(rule.name, month=rule.month, day=rule.day).dates(start, end))
    return out


def calendar_features(ts, lon, horizon_h) -> dict[str, np.ndarray]:
    """Cyclic local-solar-time features, US-holiday flag and the horizon. `ts` epoch seconds (UTC)."""
    ts = np.atleast_1d(np.asarray(ts, dtype=float))
    lon = np.broadcast_to(np.asarray(lon, dtype=float), ts.shape)
    local = pd.DatetimeIndex(pd.to_datetime(ts + lon * 240.0, unit="s"))
    hour = local.hour.to_numpy() + local.minute.to_numpy() / 60.0
    dow, month = local.dayofweek.to_numpy(), local.month.to_numpy()
    days = local.normalize()
    return {
        "hour_sin": np.sin(2 * np.pi * hour / 24), "hour_cos": np.cos(2 * np.pi * hour / 24),
        "dow_sin": np.sin(2 * np.pi * dow / 7), "dow_cos": np.cos(2 * np.pi * dow / 7),
        "month_sin": np.sin(2 * np.pi * (month - 1) / 12), "month_cos": np.cos(2 * np.pi * (month - 1) / 12),
        "holiday": days.isin(_holiday_days()).astype(float),
        "horizon_h": np.broadcast_to(np.asarray(horizon_h, dtype=float), ts.shape).copy(),
    }


def zone_columns(zones) -> dict[str, np.ndarray]:
    zones = np.asarray(zones)
    return {f"zone_{z}": (zones == z).astype(float) for z in data.ZONES}


def build_flight_features(raw: pd.DataFrame) -> pd.DataFrame:
    """Raw per-flight columns -> model features. Raw columns: ts_dep, horizon_h, o_lon, o_tier, d_tier, o_zone, distance_km,
    crs_elapsed_min, er_*, and o_/d_ prefixed obs_*/fc_* columns. Raises KeyError if one is missing."""
    cols = {c: raw[c].to_numpy(dtype=float) for c in [f"o_{c}" for c in OBS + FC] + [f"d_{c}" for c in OBS + FC] + ENROUTE + ["distance_km", "crs_elapsed_min"]}
    cols["tier_o"], cols["tier_d"] = raw["o_tier"].to_numpy(dtype=float), raw["d_tier"].to_numpy(dtype=float)
    cols.update(calendar_features(raw["ts_dep"], raw["o_lon"], raw["horizon_h"]))
    cols.update(zone_columns(raw["o_zone"]))
    return pd.DataFrame({c: cols[c] for c in FLIGHT_FEATURES}, index=raw.index)


class LGBModel:
    """LightGBM regressor on minutes (L2). Non-negative predictions; `contrib` is LightGBM pred_contrib (additive, in minutes)."""

    PARAMS = {"objective": "regression", "learning_rate": 0.05, "num_leaves": 31, "min_data_in_leaf": 100, "feature_fraction": 0.8,
              "bagging_fraction": 0.8, "bagging_freq": 1, "lambda_l2": 5.0, "verbosity": -1, "seed": 7, "deterministic": True, "force_row_wise": True,
              "num_threads": 1}  # 1 thread: fastest for small fits and live predictions; train_all raises it for real runs

    def __init__(self, feature_names, params: dict | None = None):
        self.feature_names = list(feature_names)
        self.params = {**self.PARAMS, **(params or {})}
        self.booster: lgb.Booster | None = None

    def _x(self, X: pd.DataFrame) -> np.ndarray:
        return X[self.feature_names].to_numpy(dtype=float)

    def fit(self, X: pd.DataFrame, y, weight=None, val=None, rounds: int = 300) -> "LGBModel":
        train = lgb.Dataset(self._x(X), label=np.asarray(y, dtype=float), weight=weight, feature_name=self.feature_names, free_raw_data=False)
        valid, callbacks = [], []
        if val is not None:  # (X, y) or (X, y, weight): early stopping on a time-ordered inner holdout
            vx, vy = val[0], val[1]
            valid = [lgb.Dataset(self._x(vx), label=np.asarray(vy, dtype=float), weight=val[2] if len(val) > 2 else None, reference=train)]
            callbacks = [lgb.early_stopping(25, verbose=False)]
        self.booster = lgb.train(self.params, train, num_boost_round=rounds, valid_sets=valid, callbacks=callbacks)
        return self

    def raw_predict(self, X: pd.DataFrame) -> np.ndarray:
        return self.booster.predict(self._x(X), num_threads=1)

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        return np.maximum(self.raw_predict(X), 0.0)

    def contrib(self, X: pd.DataFrame) -> np.ndarray:
        """(n, F + 1): per-feature contributions in minutes, last column is the bias; rows sum to raw_predict."""
        return self.booster.predict(self._x(X), pred_contrib=True, num_threads=1)

    def top_drivers(self, X: pd.DataFrame, k: int = 3) -> list[list[tuple[str, float]]]:
        c = self.contrib(X)[:, :-1]
        out = []
        for row in c:
            order = np.argsort(-np.abs(row))[:k]
            out.append([(self.feature_names[i], float(row[i])) for i in order])
        return out

    def save(self, path) -> None:
        path = Path(path)
        path.mkdir(parents=True, exist_ok=True)
        self.booster.save_model(str(path / "model.txt"))
        (path / "meta.json").write_text(json.dumps({"feature_names": self.feature_names, "params": self.params}), encoding="utf-8")

    @classmethod
    def load(cls, path) -> "LGBModel":
        path = Path(path)
        meta = json.loads((path / "meta.json").read_text(encoding="utf-8"))
        m = cls(meta["feature_names"], meta["params"])
        m.booster = lgb.Booster(model_file=str(path / "model.txt"))
        return m
