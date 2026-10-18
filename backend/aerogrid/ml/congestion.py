"""Airport congestion (LightGBM): airport-hour NAS-type delay -> 0..100 stress index.

Train/serve parity: features are only what exists live at issue time (typical scheduled load, METAR obs, forecast, calendar, static).
NAS status and ADS-B counts are NOT inputs (no free history; NAS is the scorecard's ground truth). `SchedProfile` is built from
scheduled counts of the TRAIN block only, so it carries no delay information.
"""
from __future__ import annotations

import json
import math
from pathlib import Path

import numpy as np
import pandas as pd

from . import weather as wx

HIST_FEATURES = [f"hp_{t}_{lvl}" for t in ("nas", "dep", "late") for lvl in ("ap", "how")]  # HistPrior columns
AIRPORT_FEATURES = ["sched_deps", "sched_arrs"] + HIST_FEATURES + wx.OBS + wx.FC + wx.CAL + ["tier"] + wx.ZONE_COLS  # congestion LightGBM
GCN_FEATURES = [f for f in AIRPORT_FEATURES if f not in HIST_FEATURES]  # the GCN and its no-graph baseline: the prior did not help them (docs/phase2-design.md, 8)
_WEEK = 7 * 86400


def hour_of_week(ts, lon) -> np.ndarray:
    """0..167 in local solar time (UTC + lon/15), Monday 00:00 = 0."""
    local = np.asarray(ts, dtype=float) + np.asarray(lon, dtype=float) * 240.0
    day = np.floor(local / 86400).astype(int)
    hour = ((local % 86400) // 3600).astype(int)
    return ((day + 3) % 7) * 24 + hour  # epoch day 0 was a Thursday


class SchedProfile:
    """Typical scheduled departures / arrivals per airport and hour-of-week (mean per week), with a per-tier fallback."""

    def __init__(self):
        self.deps: dict[str, np.ndarray] = {}
        self.arrs: dict[str, np.ndarray] = {}
        self.tier_deps: dict[int, np.ndarray] = {}
        self.tier_arrs: dict[int, np.ndarray] = {}

    def fit(self, flights: pd.DataFrame, lons: dict, tiers: dict) -> "SchedProfile":
        ts_all = np.concatenate([flights["ts_dep"].to_numpy(), flights["ts_arr"].to_numpy()])
        weeks = max(1, math.ceil((ts_all.max() - ts_all.min() + 3600) / _WEEK))
        for kind, col, key in (("deps", "ts_dep", "origin"), ("arrs", "ts_arr", "dest")):
            how = hour_of_week(flights[col].to_numpy(), flights[key].map(lons).to_numpy(dtype=float))
            counts = pd.crosstab(flights[key].to_numpy(), how).reindex(columns=range(168), fill_value=0)
            setattr(self, kind, {a: counts.loc[a].to_numpy(dtype=float) / weeks for a in counts.index})
        for tier in sorted(set(tiers.values())):
            members = [a for a in self.deps if tiers.get(a) == tier]
            if members:
                self.tier_deps[tier] = np.mean([self.deps[a] for a in members], axis=0)
                self.tier_arrs[tier] = np.mean([self.arrs.get(a, np.zeros(168)) for a in members], axis=0)
        return self

    def lookup(self, airports, tiers, ts, lons) -> tuple[np.ndarray, np.ndarray]:
        how = hour_of_week(ts, lons)
        deps, arrs = np.zeros(len(how)), np.zeros(len(how))
        for i, (a, t) in enumerate(zip(airports, tiers)):
            d = self.deps.get(a, self.tier_deps.get(int(t)))
            r = self.arrs.get(a, self.tier_arrs.get(int(t)))
            deps[i] = 0.0 if d is None else d[how[i]]
            arrs[i] = 0.0 if r is None else r[how[i]]
        return deps, arrs

    def save(self, path) -> None:
        pack = lambda d: {str(k): v.tolist() for k, v in d.items()}  # noqa: E731
        Path(path).write_text(json.dumps({"deps": pack(self.deps), "arrs": pack(self.arrs), "tier_deps": pack(self.tier_deps), "tier_arrs": pack(self.tier_arrs)}), encoding="utf-8")

    @classmethod
    def load(cls, path) -> "SchedProfile":
        raw = json.loads(Path(path).read_text(encoding="utf-8"))
        p = cls()
        arr = lambda d, conv=str: {conv(k): np.asarray(v) for k, v in d.items()}  # noqa: E731
        p.deps, p.arrs = arr(raw["deps"]), arr(raw["arrs"])
        p.tier_deps, p.tier_arrs = arr(raw["tier_deps"], int), arr(raw["tier_arrs"], int)
        return p


class HistPrior:
    """Historical mean of the airport-hour targets by airport and by airport x hour-of-week (flight-weighted, shrunk).

    This is the "airport identity" the models otherwise lack (no node-ID features, by design): a train-only statistic, so it exists
    live exactly as in training. Estimates are shrunk down a chain, so thin cells never stand alone:

        global -> tier -> airport -> airport x hour-of-week  (the last parent is airport level x the tier's hour-of-week shape)

    An airport the table has never seen (any non-US airport) gets its tier's estimates. For TRAINING rows use `oof_lookup`: each row's
    value comes from the table fitted without that row's fold (e.g. its week), so a row never sees its own target; rows outside the fit
    window use `lookup`, which is exactly what the live app does.
    """

    TARGETS = ("y_nas", "y_dep", "y_late")
    SHRINK = 60.0  # pseudo-flights pulling a cell toward its parent; fixed in advance, not tuned

    def __init__(self, shrink: float = SHRINK):
        self.shrink = float(shrink)
        self.airports: list[str] = []
        self.tiers: list[int] = []
        self.tier_of: np.ndarray = np.zeros(0, dtype=int)  # per fitted airport: index into self.tiers
        self.tables: dict[str, dict[str, np.ndarray]] = {}  # target -> {ap (A,), how (A,168), tier_ap (T,), tier_how (T,168), glob ()}
        self._sums: dict[str, tuple[np.ndarray, np.ndarray]] = {}
        self._fold_sums: dict[str, dict[int, tuple[np.ndarray, np.ndarray]]] = {}

    # -- fitting ------------------------------------------------------------------------------------------------
    def _accumulate(self, a_idx, how, n, y) -> tuple[np.ndarray, np.ndarray]:
        ok = (n > 0) & np.isfinite(y)
        flat = a_idx[ok] * 168 + how[ok]
        size = len(self.airports) * 168
        w = n[ok].astype(float)
        return (np.bincount(flat, weights=w * y[ok], minlength=size).reshape(-1, 168), np.bincount(flat, weights=w, minlength=size).reshape(-1, 168))

    def _estimate(self, S: np.ndarray, W: np.ndarray) -> dict[str, np.ndarray]:
        k, n_tier = self.shrink, len(self.tiers)
        glob = float(S.sum() / W.sum()) if W.sum() > 0 else 0.0
        St, Wt = np.zeros((n_tier, 168)), np.zeros((n_tier, 168))
        np.add.at(St, self.tier_of, S)
        np.add.at(Wt, self.tier_of, W)
        tier_ap = (St.sum(1) + k * glob) / (Wt.sum(1) + k)
        tier_how = (St + k * tier_ap[:, None]) / (Wt + k)
        shape = np.where(tier_ap[:, None] > 1e-9, tier_how / np.maximum(tier_ap[:, None], 1e-9), 1.0)
        ap = (S.sum(1) + k * tier_ap[self.tier_of]) / (W.sum(1) + k)
        how = (S + k * ap[:, None] * shape[self.tier_of]) / (W + k)
        return {"ap": ap, "how": how, "tier_ap": tier_ap, "tier_how": tier_how, "glob": np.asarray(glob)}

    def fit(self, airports, tiers, how, n, ys: dict, folds=None) -> "HistPrior":
        """airports / tiers / how (0..167) / n (flights) per airport-hour row; ys: target name -> per-row mean. `folds` (an int per
        row) is only needed for `oof_lookup`."""
        airports, how, n = np.asarray(airports), np.asarray(how, dtype=int), np.asarray(n, dtype=float)
        self.airports = sorted(set(airports.tolist()))
        a_idx = pd.Index(self.airports).get_indexer(airports)
        tier_by_airport = dict(zip(airports.tolist(), np.asarray(tiers, dtype=int).tolist()))
        self.tiers = sorted(set(tier_by_airport.values()))
        self.tier_of = np.array([self.tiers.index(tier_by_airport[a]) for a in self.airports], dtype=int)
        folds = None if folds is None else np.asarray(folds, dtype=int)
        self._sums, self._fold_sums, self.tables = {}, {}, {}
        for t in self.TARGETS:
            y = np.asarray(ys[t], dtype=float)
            self._sums[t] = self._accumulate(a_idx, how, n, y)
            self.tables[t] = self._estimate(*self._sums[t])
            if folds is not None:
                self._fold_sums[t] = {int(f): self._accumulate(a_idx[folds == f], how[folds == f], n[folds == f], y[folds == f]) for f in np.unique(folds)}
        return self

    # -- reading ------------------------------------------------------------------------------------------------
    def _columns(self, airports, tiers, how, tables_for) -> dict[str, np.ndarray]:
        """tables_for(target) -> table dict. Seen airports read their own rows, unseen ones their tier's (global if the tier is unseen too)."""
        airports, how = np.asarray(airports), np.asarray(how, dtype=int)
        a_idx = pd.Index(self.airports).get_indexer(airports)
        t_idx = pd.Index(self.tiers).get_indexer(np.asarray(tiers, dtype=int))
        seen = a_idx >= 0
        out = {}
        for t in self.TARGETS:
            tb = tables_for(t)
            ap = np.where(seen, tb["ap"][np.maximum(a_idx, 0)], np.where(t_idx >= 0, tb["tier_ap"][np.maximum(t_idx, 0)], tb["glob"]))
            hw = np.where(seen, tb["how"][np.maximum(a_idx, 0), how], np.where(t_idx >= 0, tb["tier_how"][np.maximum(t_idx, 0), how], tb["glob"]))
            name = t[2:]
            out[f"hp_{name}_ap"], out[f"hp_{name}_how"] = ap, hw
        return out

    def lookup(self, airports, tiers, ts, lons) -> dict[str, np.ndarray]:
        """Feature columns hp_{nas,dep,late}_{ap,how} from the fitted table (live path; also every training row outside the fit window)."""
        return self._columns(airports, tiers, hour_of_week(ts, lons), lambda t: self.tables[t])

    def oof_lookup(self, airports, tiers, ts, lons, folds) -> dict[str, np.ndarray]:
        """Same columns for rows that were in `fit`, each from the table fitted without that row's fold."""
        if not self._fold_sums:
            raise ValueError("fit(..., folds=...) is needed for out-of-fold values")
        folds, how = np.asarray(folds, dtype=int), hour_of_week(ts, lons)
        out: dict[str, np.ndarray] = {}
        for f in np.unique(folds):
            m = folds == f
            cols = self._columns(np.asarray(airports)[m], np.asarray(tiers)[m], how[m],
                                 lambda t, f=f: self._estimate(self._sums[t][0] - self._fold_sums[t][int(f)][0], self._sums[t][1] - self._fold_sums[t][int(f)][1]))
            for c, v in cols.items():
                out.setdefault(c, np.zeros(len(folds)))[m] = v
        return out

    # -- artifact -----------------------------------------------------------------------------------------------
    def save(self, path) -> None:
        pack = {t: {k: np.asarray(v).tolist() for k, v in tb.items()} for t, tb in self.tables.items()}
        Path(path).write_text(json.dumps({"shrink": self.shrink, "airports": self.airports, "tiers": self.tiers, "tier_of": self.tier_of.tolist(), "tables": pack}), encoding="utf-8")

    @classmethod
    def load(cls, path) -> "HistPrior":
        raw = json.loads(Path(path).read_text(encoding="utf-8"))
        p = cls(raw["shrink"])
        p.airports, p.tiers, p.tier_of = raw["airports"], raw["tiers"], np.asarray(raw["tier_of"], dtype=int)
        p.tables = {t: {k: np.asarray(v, dtype=float) for k, v in tb.items()} for t, tb in raw["tables"].items()}
        return p



def build_airport_features(raw: pd.DataFrame) -> pd.DataFrame:
    """Raw airport-hour columns -> model features. Raw: tier, lon, zone, ts, horizon_h, sched_deps, sched_arrs, hp_*, obs_*, fc_*."""
    cols = {c: raw[c].to_numpy(dtype=float) for c in ["sched_deps", "sched_arrs"] + HIST_FEATURES + wx.OBS + wx.FC}
    cols["tier"] = raw["tier"].to_numpy(dtype=float)
    cols.update(wx.calendar_features(raw["ts"], raw["lon"], raw["horizon_h"]))
    cols.update(wx.zone_columns(raw["zone"]))
    return pd.DataFrame({c: cols[c] for c in AIRPORT_FEATURES}, index=raw.index)


class CongestionModel:
    def __init__(self, params: dict | None = None):
        self.gbm = wx.LGBModel(AIRPORT_FEATURES, params)
        self.quantiles: np.ndarray | None = None

    def fit(self, X: pd.DataFrame, y, n, val=None, rounds: int = 300) -> "CongestionModel":
        """Rows with n == 0 flights are ignored; weight = min(n, 20) / 20."""
        n = np.asarray(n)
        keep = n > 0
        y = np.asarray(y, dtype=float)
        if val is not None:  # (X_val, y_val, n_val)
            vx, vy, vn = val[0], np.asarray(val[1], dtype=float), np.asarray(val[2])
            vk = vn > 0
            val = (vx[vk], vy[vk], np.minimum(vn[vk], 20) / 20.0)
        self.gbm.fit(X[keep], y[keep], weight=np.minimum(n[keep], 20) / 20.0, val=val, rounds=rounds)
        self.quantiles = np.maximum.accumulate(np.percentile(self.gbm.predict(X[keep]), np.linspace(0, 100, 101)))
        return self

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        return self.gbm.predict(X)

    def stress_from_pred(self, pred) -> np.ndarray:
        """Train-set percentile (0..100) of the predicted delay."""
        return np.interp(np.asarray(pred, dtype=float), self.quantiles, np.linspace(0, 100, 101))

    def stress(self, X: pd.DataFrame) -> np.ndarray:
        return self.stress_from_pred(self.predict(X))

    def save(self, path) -> None:
        path = Path(path)
        self.gbm.save(path)
        (path / "stress_quantiles.json").write_text(json.dumps(self.quantiles.tolist()), encoding="utf-8")

    @classmethod
    def load(cls, path) -> "CongestionModel":
        path = Path(path)
        m = cls()
        m.gbm = wx.LGBModel.load(path)
        m.quantiles = np.asarray(json.loads((path / "stress_quantiles.json").read_text(encoding="utf-8")))
        return m
