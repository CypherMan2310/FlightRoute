"""Stacked ensemble: a non-negative linear meta-learner fitted per weather zone on out-of-fold component predictions.

    delay_minutes = baseline + sum_k w_k * component_k        (w_k >= 0, baseline >= 0)

Because it is additive, the cause breakdown is exact: each cause is w_k * component_k and the parts plus the baseline sum
to the prediction. Zones with fewer than `min_rows` out-of-fold rows borrow the pooled weights (flagged `pooled`).
The carrier component is a shrunk lookup table (decision 1): mean CarrierDelay + SecurityDelay by airline x origin x month.
"""
from __future__ import annotations

import json
from functools import lru_cache
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import sparse
from scipy.optimize import nnls

from .. import data
from . import bird, gcn, routing
from . import congestion as cg
from . import conformal as cf
from . import weather as wx

COMPONENTS = ("weather", "congestion", "cascade", "route", "bird", "carrier")


def _month(ts) -> np.ndarray:
    return pd.DatetimeIndex(pd.to_datetime(np.asarray(ts, dtype=float), unit="s")).month.to_numpy()


class CarrierModel:
    """Mean carrier + security delay by carrier x origin x month, with hierarchical shrinkage.

    Each level's estimate is (n * cell_mean + shrink * parent_estimate) / (n + shrink), going global -> carrier -> carrier x origin ->
    carrier x origin x month. Cell means of a heavy-tailed delay are noisy, so small cells are pulled toward their parent instead
    of being used (or dropped) at a hard count threshold. Fitted on train only.
    """

    LEVELS = ("carrier x origin x month", "carrier x origin", "carrier", "global")

    def __init__(self, shrink: float = 100.0):
        self.shrink = shrink
        self.cells: dict[str, dict[str, tuple[float, int]]] = {lvl: {} for lvl in self.LEVELS[:3]}
        self.global_mean = 0.0

    def fit(self, df: pd.DataFrame) -> "CarrierModel":
        v = df["carrier_delay"].to_numpy(dtype=float) + df["security_delay"].to_numpy(dtype=float)
        d = pd.DataFrame({"c": df["carrier"].to_numpy(), "o": df["origin"].to_numpy(), "m": _month(df["ts_dep"]), "v": v})
        self.global_mean = float(d["v"].mean())
        for lvl, keys in zip(self.LEVELS[:3], (["c", "o", "m"], ["c", "o"], ["c"])):
            g = d.groupby(keys)["v"].agg(["mean", "size"])
            self.cells[lvl] = {"|".join(map(str, k if isinstance(k, tuple) else (k,))): (float(r["mean"]), int(r["size"])) for k, r in g.iterrows()}
        return self

    def _lookup(self, carrier, origin, month) -> tuple[str, float, int]:
        """(deepest level with data, shrunk estimate, n of that cell)."""
        est, level, n = self.global_mean, "global", 0
        for lvl, key in zip(self.LEVELS[2::-1], (f"{carrier}", f"{carrier}|{origin}", f"{carrier}|{origin}|{month}")):
            cell = self.cells[lvl].get(key)
            if cell is None:
                break  # no data at this depth, so none deeper either
            est = (cell[1] * cell[0] + self.shrink * est) / (cell[1] + self.shrink) if cell[1] + self.shrink > 0 else cell[0]
            level, n = lvl, cell[1]
        return level, est, n

    def predict(self, df: pd.DataFrame) -> np.ndarray:
        months = _month(df["ts_dep"])
        return np.array([self._lookup(c, o, m)[1] for c, o, m in zip(df["carrier"], df["origin"], months)])

    def explain(self, carrier: str, origin: str, ts_dep: float) -> dict:
        level, mean, n = self._lookup(carrier, origin, int(_month([ts_dep])[0]))
        return {"level": level, "mean_min": mean, "n": n}

    def save(self, path) -> None:
        Path(path).write_text(json.dumps({"shrink": self.shrink, "global_mean": self.global_mean, "cells": self.cells}), encoding="utf-8")

    @classmethod
    def load(cls, path) -> "CarrierModel":
        d = json.loads(Path(path).read_text(encoding="utf-8"))
        m = cls(d["shrink"])
        m.global_mean = d["global_mean"]
        m.cells = {lvl: {k: (v[0], v[1]) for k, v in cells.items()} for lvl, cells in d["cells"].items()}
        return m


def _solve(c: np.ndarray, y: np.ndarray) -> dict:
    coef, _ = nnls(np.column_stack([c, np.ones(len(c))]), y)
    return {"weights": {k: float(w) for k, w in zip(COMPONENTS, coef[:-1])}, "baseline": float(coef[-1]), "n": int(len(y))}


class Ensemble:
    def __init__(self):
        self.zones: dict[str, dict] = {}
        self.pooled: dict = {}

    def fit(self, c: pd.DataFrame, y, zones, min_rows: int = 1000) -> "Ensemble":
        cm, y, zones = c[list(COMPONENTS)].to_numpy(dtype=float), np.asarray(y, dtype=float), np.asarray(zones)
        self.pooled = _solve(cm, y)
        self.zones = {}
        for z in np.unique(zones):
            m = zones == z
            if m.sum() >= min_rows:
                self.zones[str(z)] = {**_solve(cm[m], y[m]), "pooled": False}
            else:
                self.zones[str(z)] = {**self.pooled, "n": int(m.sum()), "pooled": True}
        return self

    def _params(self, zone: str) -> dict:
        return self.zones.get(str(zone), {**self.pooled, "pooled": True})

    def predict(self, c: pd.DataFrame, zones) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """(total, parts[n, 6], baseline[n]); total == parts.sum(1) + baseline exactly."""
        cm = c[list(COMPONENTS)].to_numpy(dtype=float)
        parts, base = np.zeros_like(cm), np.zeros(len(cm))
        for i, z in enumerate(np.asarray(zones)):
            p = self._params(z)
            parts[i] = cm[i] * np.array([p["weights"][k] for k in COMPONENTS])
            base[i] = p["baseline"]
        return parts.sum(axis=1) + base, parts, base

    def breakdown(self, c_row, zone: str) -> dict:
        row = pd.DataFrame([dict(c_row)])
        total, parts, base = self.predict(row, [zone])
        p = self._params(zone)
        return {
            "total_min": float(total[0]),
            "baseline_min": float(base[0]),
            "causes": {k: float(v) for k, v in zip(COMPONENTS, parts[0])},
            "weights": p["weights"],
            "pooled_weights": bool(p["pooled"]),
            "zone_rows": p["n"],
        }

    def save(self, path) -> None:
        Path(path).write_text(json.dumps({"zones": self.zones, "pooled": self.pooled}), encoding="utf-8")

    @classmethod
    def load(cls, path) -> "Ensemble":
        d = json.loads(Path(path).read_text(encoding="utf-8"))
        e = cls()
        e.zones, e.pooled = d["zones"], d["pooled"]
        return e


# ---- runtime bundle: load the trained artifacts and run them (no network, no torch) -------------------------------

_SIZE = {1: 9.0, 2: 5.0, 3: 2.5}
MAX_GCN_MIN = 180.0


def nearest_horizon(hours: float) -> int:
    """The models were trained with issue-time horizons of 2, 6 and 24 hours."""
    return 2 if hours <= 4 else 6 if hours <= 12 else 24


@lru_cache(maxsize=1)
def world_operator() -> sparse.csr_matrix:
    """In-flow operator over the world routes graph. Edge weight = size_i * size_j from the hub tiers (no real traffic counts exist
    for the world graph yet); the GCN weights carry no node identity, so they run on this graph."""
    g = data.routes_graph()
    size = np.array([_SIZE[a.tier] for a in data.load_airports()])
    tri = g.adjacency.tocoo()
    return gcn.operator_inflow(sparse.coo_matrix((size[tri.row] * size[tri.col], (tri.row, tri.col)), shape=g.adjacency.shape))


class Bundle:
    def __init__(self, path):
        self.path = Path(path)
        self.card = json.loads((self.path / "model_card.json").read_text(encoding="utf-8"))
        meta = json.loads((self.path / "nodes.json").read_text(encoding="utf-8"))
        self.nodes, self.crs_fit = meta["nodes"], meta["crs_fit"]
        self.gcn, self.gcn_meta = gcn.GCN.load(self.path / "gcn.npz")
        assert self.gcn_meta["feature_names"] == cg.GCN_FEATURES, "artifact features do not match the code"
        self.congestion = cg.CongestionModel.load(self.path / "congestion")
        assert self.congestion.gbm.feature_names == cg.AIRPORT_FEATURES, "artifact features do not match the code"
        self.weather = wx.LGBModel.load(self.path / "weather")
        self.carrier = CarrierModel.load(self.path / "carrier.json")
        self.sched = cg.SchedProfile.load(self.path / "sched.json")
        self.hist = cg.HistPrior.load(self.path / "hist.json")
        self.ensemble = Ensemble.load(self.path / "ensemble.json")
        self.conformal = cf.SplitConformal.load(self.path / "conformal.json")

    @classmethod
    def load_or_none(cls, path) -> "Bundle | None":
        try:
            if json.loads((Path(path) / "model_card.json").read_text(encoding="utf-8")).get("status") != "trained":
                return None
            return cls(path)
        except (OSError, ValueError, KeyError, AssertionError):
            return None

    @classmethod
    def load_error(cls, path) -> "str | None":
        """Why the artifacts in `path` cannot be used (missing, untrained or incompatible), or None if they load."""
        try:
            if json.loads((Path(path) / "model_card.json").read_text(encoding="utf-8")).get("status") != "trained":
                return "model card status is not 'trained'"
            cls(path)
            return None
        except FileNotFoundError as exc:
            return f"missing artifact: {Path(exc.filename).name}"
        except KeyError as exc:
            return f"artifact is missing the field {exc}: retrain with the current code"
        except (OSError, ValueError, AssertionError) as exc:
            return f"artifacts could not be loaded: {exc}"

    @property
    def trained_on(self) -> str:
        return self.card["trained_on"]

    @property
    def status(self) -> str:
        return self.card["status"]

    def scheduled_minutes(self, distance_km: float) -> float:
        """Scheduled block time estimate, the same linear fit of the training flights."""
        return max(20.0, self.crs_fit[0] * distance_km + self.crs_fit[1])

    def world_frame(self, snap: dict) -> pd.DataFrame:
        """Raw airport-hour frame (one row per world airport) from a live snapshot {ts, horizon_h, obs{iata: canonical}, fc{iata: canonical}}."""
        airports = data.load_airports()
        ts = float(snap["ts"])
        month = int(pd.Timestamp(ts, unit="s", tz="UTC").month)
        raw = pd.DataFrame({
            "airport": [a.iata for a in airports], "tier": [a.tier for a in airports], "lon": [a.lon for a in airports],
            "zone": [data.weather_zone(a.lat, a.lon, month) for a in airports], "ts": ts, "horizon_h": int(snap["horizon_h"]),
        })
        for k in wx.OBS:
            raw[k] = [snap["obs"].get(a.iata, wx.NEUTRAL_OBS)[k] for a in airports]
        for k in wx.FC:
            raw[k] = [snap["fc"].get(a.iata, wx.NEUTRAL_FC)[k] for a in airports]
        raw["sched_deps"], raw["sched_arrs"] = self.sched.lookup(raw["airport"], raw["tier"], raw["ts"], raw["lon"])
        for c, v in self.hist.lookup(raw["airport"], raw["tier"], raw["ts"], raw["lon"]).items():
            raw[c] = v  # airport identity from the train-only table; airports it has never seen (non-US) get their tier's values
        return raw

    def score_world(self, raw: pd.DataFrame) -> dict:
        x = cg.build_airport_features(raw)
        cong = self.congestion.predict(x)
        gcn_out = np.clip(self.gcn.predict(world_operator(), x[cg.GCN_FEATURES].to_numpy(dtype=float)), 0.0, MAX_GCN_MIN)  # world graph is out of the training distribution: keep minutes physical
        return {"codes": list(raw["airport"]), "features": x, "cong_pred": cong, "stress": self.congestion.stress_from_pred(cong), "gcn": gcn_out}

    def predict_flights(self, rows: pd.DataFrame, score: dict) -> dict:
        index = {c: i for i, c in enumerate(score["codes"])}
        idx = np.array([index[o] for o in rows["origin"]])
        xf = wx.build_flight_features(rows)
        comp = pd.DataFrame({
            "weather": self.weather.predict(xf), "congestion": score["cong_pred"][idx], "cascade": score["gcn"][idx, 1],
            "route": rows["route_extra_min"].to_numpy(dtype=float), "bird": rows["bird_index"].to_numpy(dtype=float), "carrier": self.carrier.predict(rows),
        })[list(COMPONENTS)]
        total, parts, base = self.ensemble.predict(comp, rows["o_zone"].to_numpy())
        lo, hi = self.conformal.interval(total)
        return {"components": comp, "causes": pd.DataFrame(parts, columns=list(COMPONENTS)), "baseline": base, "total": total, "lo": lo, "hi": hi, "flight_features": xf, "origin_index": idx}


def build_flight_row(bundle: "Bundle", origin, dest, carrier, dep_ts, horizon_h, o_obs, d_obs, o_fc, d_fc, er_fc, er_wind) -> pd.DataFrame:
    """One raw flight row with the same columns the training tables have. er_fc: three canonical fc dicts (25/50/75 %);
    er_wind: three (jet speed kt, direction-from deg)."""
    o, d = data.get_airport(origin), data.get_airport(dest)
    dist = data.great_circle_km(o.lat, o.lon, d.lat, d.lon)
    crs = bundle.scheduled_minutes(dist)
    month = int(pd.Timestamp(dep_ts, unit="s", tz="UTC").month)
    ts_arr = dep_ts + crs * 60
    row = {
        "origin": o.iata, "dest": d.iata, "carrier": carrier, "ts_dep": float(dep_ts), "ts_arr": ts_arr, "ts_hour": float(np.floor(dep_ts / 3600) * 3600),
        "crs_elapsed_min": crs, "distance_km": dist, "horizon_h": int(horizon_h), "o_lon": o.lon, "o_tier": o.tier, "d_tier": d.tier,
        "o_zone": data.weather_zone(o.lat, o.lon, month),
        "er_max_gust_kt": max(f["fc_gust_kt"] for f in er_fc), "er_max_precip_mm": max(f["fc_precip_mm"] for f in er_fc),
        "er_max_cape_jkg": max(f["fc_cape_jkg"] for f in er_fc), "er_max_cloud_low_pct": max(f["fc_cloud_low_pct"] for f in er_fc),
        "route_extra_min": routing.wind_extra_minutes(o.lat, o.lon, d.lat, d.lon, er_wind)["extra_min"],
        "bird_index": float(bird.flight_bird_index(o.lat, o.lon, dep_ts, d.lat, d.lon, ts_arr)),
    }
    for k, v in {**o_obs, **o_fc}.items():
        row[f"o_{k}"] = v
    for k, v in {**d_obs, **d_fc}.items():
        row[f"d_{k}"] = v
    return pd.DataFrame([row])
