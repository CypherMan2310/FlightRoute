"""Train every model, save artifacts and model_card.json (with trained_on). Offline only.

    python -m aerogrid.train.train_all --source synthetic --out <dir> [--quick]
    python -m aerogrid.train.train_all --source real --months 30 --out <dir>

Protocol (docs/phase2-design.md): time-ordered blocks train / gap / calib / gap / test, never shuffled. Every statistic is fitted on
train only. Stacker inputs are forward-chaining out-of-fold predictions on train; conformal intervals use the calib block only;
reported numbers use the test block only, each with its n and a block-bootstrap (by day) 95% CI, next to baselines.
"""
from __future__ import annotations

import argparse
import json
import sys
import tempfile
from dataclasses import dataclass, replace
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import sparse

from ..ml import bird as bird_mod
from ..ml import congestion as cg
from ..ml import conformal as cf
from ..ml import ensemble as en
from ..ml import gcn
from ..ml import routing
from ..ml import weather as wx
from . import Tables

GAP_S = 86400
MIN_ZONE_TEST = 1000
TARGET_COVERAGE = 0.8


@dataclass
class Config:
    n_folds: int = 4
    gcn_hidden: int = 32
    gcn_epochs: int = 80
    gcn_lr: float = 3e-3
    gcn_batch: int = 32
    rounds: int = 1000  # upper bound: early stopping on the inner holdout decides
    bootstrap: int = 1000
    min_zone_rows: int = 1000
    threads: int = 1


CONFIGS = {"full": Config(), "quick": Config(n_folds=2, gcn_hidden=8, gcn_epochs=15, gcn_lr=5e-3, rounds=40, bootstrap=100)}


def _iso(ts: float) -> str:
    return datetime.fromtimestamp(float(ts), tz=timezone.utc).strftime("%Y-%m-%d")


def metrics(y, p, *, days=None, w=None, f1_key="f1_gt15", boot=1000, seed=0) -> dict:
    """MAE, RMSE, F1 for 'value > 15 min', n, and a block-bootstrap-over-days 95% CI of the MAE."""
    y, p = np.asarray(y, dtype=float), np.asarray(p, dtype=float)
    err = np.abs(p - y)
    out = {"n": int(len(y)), "mae": float(err.mean()), "rmse": float(np.sqrt(((p - y) ** 2).mean()))}
    tp, fp, fn = int(((y > 15) & (p > 15)).sum()), int(((y <= 15) & (p > 15)).sum()), int(((y > 15) & (p <= 15)).sum())
    out[f1_key] = float(2 * tp / max(2 * tp + fp + fn, 1))
    if w is not None:
        out["mae_weighted"] = float((err * w).sum() / max(float(np.sum(w)), 1e-12))
        out["rmse_weighted"] = float(np.sqrt((((p - y) ** 2) * w).sum() / max(float(np.sum(w)), 1e-12)))
    if days is not None and len(y) > 0:
        d = np.unique(days, return_inverse=True)[1]
        sums, cnt = np.bincount(d, weights=err), np.bincount(d)
        idx = np.random.default_rng(seed).integers(0, len(sums), (boot, len(sums)))
        stat = sums[idx].sum(axis=1) / cnt[idx].sum(axis=1)
        out["mae_ci95"] = [float(np.percentile(stat, 2.5)), float(np.percentile(stat, 97.5))]
    return out


def vs_baseline(y, p, base, *, days, boot=1000, seed=0) -> dict:
    """RMSE(model) / RMSE(baseline) with a paired block-bootstrap (by day) 95% CI. `better` only when the whole CI is below 1."""
    y, p, base = (np.asarray(a, dtype=float) for a in (y, p, base))
    d = np.unique(days, return_inverse=True)[1]
    num, den = np.bincount(d, weights=(p - y) ** 2), np.bincount(d, weights=(base - y) ** 2)
    idx = np.random.default_rng(seed).integers(0, len(num), (boot, len(num)))
    ratio = np.sqrt(num[idx].sum(axis=1) / np.maximum(den[idx].sum(axis=1), 1e-12))
    lo, hi = float(np.percentile(ratio, 2.5)), float(np.percentile(ratio, 97.5))
    return {"rmse_ratio": float(np.sqrt(num.sum() / max(den.sum(), 1e-12))), "ci95": [lo, hi], "better": bool(hi < 1.0), "n_days": int(len(num))}


def _guard_card(out: Path, trained_on: str, force: bool) -> None:
    card = out / "model_card.json"
    if card.exists() and not force and trained_on != "real":
        try:
            existing = json.loads(card.read_text(encoding="utf-8")).get("trained_on")
        except (ValueError, OSError):
            existing = None
        if existing == "real":
            raise RuntimeError(f"{card} is trained_on=real: refusing to overwrite it with {trained_on} output (pass force=True / --force)")


class _Chunks:
    """Forward-chaining chunks of the train period (by time)."""

    def __init__(self, train_ts: np.ndarray, n_folds: int):
        ts = np.sort(np.unique(train_ts))
        self.edges = ts[(np.arange(1, n_folds + 1) * len(ts)) // (n_folds + 1)]
        self.n_folds = n_folds

    def of(self, ts) -> np.ndarray:
        return np.searchsorted(self.edges, np.asarray(ts, dtype=float), side="right")

    def fit_limit(self, j: int) -> float:
        return float(self.edges[j - 1]) - GAP_S  # rows before this may fit the model that predicts chunk j


def _hist_mean(train: pd.DataFrame, target: str, keys: list[str], test: pd.DataFrame, w: str | None = None) -> np.ndarray:
    """Historical-mean baseline: mean target by `keys` on train (weighted by `w`), falling back to the global mean."""
    tr = train.dropna(subset=[target])
    if w:
        num = (tr[target] * tr[w]).groupby([tr[k] for k in keys]).sum()
        den = tr[w].groupby([tr[k] for k in keys]).sum()
        table, glob = num / den, float((tr[target] * tr[w]).sum() / tr[w].sum())
    else:
        table, glob = tr.groupby(keys)[target].mean(), float(tr[target].mean())
    got = table.reindex(pd.MultiIndex.from_frame(test[keys]) if len(keys) > 1 else test[keys[0]]).to_numpy()
    return np.where(np.isnan(got), glob, got)


def train(tables: Tables, out_dir, *, quick: bool = False, force: bool = False, config: Config | None = None, seed: int = 0) -> dict:
    cfg = config or CONFIGS["quick" if quick else "full"]
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    trained_on = tables.meta["trained_on"]
    _guard_card(out, trained_on, force)
    sp, nodes = tables.split, list(tables.nodes)
    n_nodes, nidx = len(nodes), {a: i for i, a in enumerate(nodes)}
    params = {"num_threads": cfg.threads}

    fl, ah = tables.flights.reset_index(drop=True).copy(), tables.airport_hour.reset_index(drop=True).copy()

    def block(ts):
        ts = np.asarray(ts, dtype=float)
        return np.select([ts < sp["train_end"], (ts >= sp["calib_start"]) & (ts < sp["calib_end"]), ts >= sp["test_start"]], ["train", "calib", "test"], "gap")

    fl["block"], ah["block"] = block(fl["ts_hour"]), block(ah["ts"])
    for b in ("train", "calib", "test"):
        if not ((fl["block"] == b).any() and (ah["block"] == b).any()):
            raise ValueError(f"the {b!r} block is empty: the window is too short for the train / gap / calib / gap / test protocol")
    lons, tiers = dict(zip(tables.airports["iata"], tables.airports["lon"])), dict(zip(tables.airports["iata"], tables.airports["tier"]))
    sched = cg.SchedProfile().fit(fl[fl["block"] == "train"], lons, tiers)
    ah["sched_deps"], ah["sched_arrs"] = sched.lookup(ah["airport"], ah["tier"], ah["ts"], ah["lon"])

    ts_unique = np.sort(ah["ts"].unique())
    assert len(ah) == len(ts_unique) * n_nodes, "airport_hour must be a complete (airport x hour) grid"
    ah["s"], ah["a"] = np.searchsorted(ts_unique, ah["ts"]), ah["airport"].map(nidx)
    fl["s"], fl["a"], fl["d"] = np.searchsorted(ts_unique, fl["ts_hour"]), fl["origin"].map(nidx), fl["dest"].map(nidx)
    chunks = _Chunks(ah.loc[ah["block"] == "train", "ts"].to_numpy(), cfg.n_folds)
    ah["chunk"], fl["chunk"] = chunks.of(ah["ts"]), chunks.of(fl["ts_hour"])
    train_days = max(1.0, (sp["train_end"] - ts_unique[0]) / 86400)
    ah_ts, fl_ts = ah["ts"].to_numpy(), fl["ts_hour"].to_numpy()
    is_tr = (ah["block"] == "train").to_numpy()

    # ---- historical prior (airport identity for congestion): leave-one-week-out inside a fit window, table lookup outside it ------
    ah["how"] = cg.hour_of_week(ah_ts, ah["lon"].to_numpy())
    ah["week"] = ((ah_ts - ts_unique[0]) // (7 * 86400)).astype(int)
    hist_y = {t: ah[t].to_numpy(dtype=float) for t in cg.HistPrior.TARGETS}

    def hist_for(fit_mask) -> tuple[cg.HistPrior, dict]:
        r = ah[fit_mask]
        prior = cg.HistPrior().fit(r["airport"], r["tier"], r["how"], r["n"], {t: v[fit_mask] for t, v in hist_y.items()}, folds=r["week"])
        cols = prior.lookup(ah["airport"], ah["tier"], ah["ts"], ah["lon"])
        for c, v in prior.oof_lookup(r["airport"], r["tier"], r["ts"], r["lon"], r["week"]).items():
            cols[c][fit_mask] = v
        return prior, cols

    hist_final, hist_cols = hist_for(is_tr)
    for c, v in hist_cols.items():
        ah[c] = v
    xa, xf = cg.build_airport_features(ah), wx.build_flight_features(fl)

    # ---- GCN over the airport graph -------------------------------------------------------------------------
    flows = sparse.coo_matrix((np.ones(int((fl["block"] == "train").sum())), (fl.loc[fl["block"] == "train", "a"], fl.loc[fl["block"] == "train", "d"])), shape=(n_nodes, n_nodes)).tocsr() / train_days
    ahat = gcn.operator_inflow(flows)
    t_count, f_count = len(ts_unique), len(cg.GCN_FEATURES)
    X = np.zeros((t_count, n_nodes, f_count))
    X[ah["s"], ah["a"]] = xa[cg.GCN_FEATURES].to_numpy()
    Y = np.zeros((t_count, n_nodes, 2))
    Y[ah["s"], ah["a"]] = np.nan_to_num(ah[["y_dep", "y_late"]].to_numpy(dtype=float))
    M = np.zeros((t_count, n_nodes))
    M[ah["s"], ah["a"]] = (ah["n"] > 0).astype(float)
    W = np.zeros((t_count, n_nodes))
    W[ah["s"], ah["a"]] = np.minimum(ah["n"], 20) / 20.0
    snap_block, snap_chunk = block(ts_unique), chunks.of(ts_unique)

    def fit_gcn(fit_mask: np.ndarray) -> gcn.GCN:
        idx = np.flatnonzero(fit_mask)
        cut = idx[int(0.85 * len(idx))]
        tr_idx = idx[(idx < cut) & (ts_unique[idx] < ts_unique[cut] - GAP_S)]
        va_idx = idx[idx >= cut]
        scaler = gcn.Standardizer().fit(X[tr_idx])
        model = gcn.GCN(f_count, hidden=cfg.gcn_hidden, seed=seed)
        model.fit(ahat, X[tr_idx], Y[tr_idx], M[tr_idx], W[tr_idx], val=(X[va_idx], Y[va_idx], M[va_idx], W[va_idx]), scaler=scaler,
                  epochs=cfg.gcn_epochs, batch=cfg.gcn_batch, lr=cfg.gcn_lr, patience=10, seed=seed)
        return model

    gcn_final = fit_gcn(snap_block == "train")
    pg = gcn_final.predict(ahat, X)  # (T, N, 2) from the final model; train chunks >= 1 are replaced by out-of-fold predictions
    for j in range(1, cfg.n_folds + 1):
        fold = fit_gcn((snap_block == "train") & (ts_unique < chunks.fit_limit(j)))
        sel = (snap_block == "train") & (snap_chunk == j)
        pg[sel] = fold.predict(ahat, X[sel])

    # ---- congestion (airport-hour LightGBM) + no-graph baselines --------------------------------------------
    def split_inner(mask, ts):
        """Time-ordered inner holdout for early stopping: the last 15% of the rows, with a gap before it."""
        cut = np.quantile(np.asarray(ts)[mask], 0.85)
        return mask & (ts < cut - GAP_S), mask & (ts >= cut)

    y_nas, n_f = ah["y_nas"].fillna(0).to_numpy(), ah["n"].to_numpy()

    def fit_cong(mask, x, features=cg.AIRPORT_FEATURES):
        fit, val = split_inner(mask, ah_ts)
        model = cg.CongestionModel(params)
        model.gbm = wx.LGBModel(features, params)
        return model.fit(x[fit], y_nas[fit], n_f[fit], val=(x[val], y_nas[val], n_f[val]), rounds=cfg.rounds)

    cong_final = fit_cong(is_tr, xa)
    cong_pred = cong_final.predict(xa)
    nohist_pred = np.full(len(ah), np.nan)  # same model without the prior, out of fold (train chunks >= 1) only; the final one is fitted below
    for j in range(1, cfg.n_folds + 1):
        fit_mask = is_tr & (ah_ts < chunks.fit_limit(j))
        sel = is_tr & (ah["chunk"].to_numpy() == j)
        xj = xa.copy()  # the prior is refitted on this fold's fit window, so the fold model never sees a table that contains its own chunk
        xj[cg.HIST_FEATURES] = pd.DataFrame(hist_for(fit_mask)[1], index=xa.index)[cg.HIST_FEATURES]
        cong_pred[sel] = fit_cong(fit_mask, xj).predict(xj[sel])
        nohist_pred[sel] = fit_cong(fit_mask, xa, cg.GCN_FEATURES).predict(xa[sel])

    # ---- weather (flight-level LightGBM), carrier lookup ----------------------------------------------------
    f_tr = (fl["block"] == "train").to_numpy()

    def fit_lgb_flights(mask, target):
        fit, val = split_inner(mask, fl_ts)
        return wx.LGBModel(wx.FLIGHT_FEATURES, params).fit(xf[fit], target[fit], val=(xf[val], target[val]), rounds=cfg.rounds)

    w_target = fl["weather_delay"].to_numpy(dtype=float)

    def fit_weather(mask):
        return fit_lgb_flights(mask, w_target)

    weather_final = fit_weather(f_tr)
    weather_pred = weather_final.predict(xf)
    carrier_final = en.CarrierModel().fit(fl[f_tr])
    carrier_pred = carrier_final.predict(fl)
    for j in range(1, cfg.n_folds + 1):
        fit_rows = f_tr & (fl["ts_hour"].to_numpy() < chunks.fit_limit(j))
        sel = f_tr & (fl["chunk"].to_numpy() == j)
        weather_pred[sel] = fit_weather(fit_rows).predict(xf[sel])
        carrier_pred[sel] = en.CarrierModel().fit(fl[fit_rows]).predict(fl[sel])

    # ---- stack ---------------------------------------------------------------------------------------------
    cong_grid = np.zeros((t_count, n_nodes))
    cong_grid[ah["s"], ah["a"]] = cong_pred
    comp = pd.DataFrame({
        "weather": weather_pred, "congestion": cong_grid[fl["s"], fl["a"]], "cascade": pg[fl["s"], fl["a"], 1],
        "route": fl["route_extra_min"].to_numpy(), "bird": fl["bird_index"].to_numpy(), "carrier": carrier_pred,
    })[list(en.COMPONENTS)]
    stack_rows = f_tr & (fl["chunk"].to_numpy() >= 1)
    y_arr = fl["arr_delay"].to_numpy(dtype=float)
    ens = en.Ensemble().fit(comp[stack_rows], y_arr[stack_rows], fl.loc[stack_rows, "o_zone"].to_numpy(), min_rows=cfg.min_zone_rows)
    total, parts, base = ens.predict(comp, fl["o_zone"].to_numpy())
    cal = (fl["block"] == "calib").to_numpy()
    conf = cf.SplitConformal(alpha=1 - TARGET_COVERAGE).fit(total[cal], y_arr[cal])

    # ---- evaluation on the test block only ---------------------------------------------------------------
    ts0 = ts_unique[0]
    te = (fl["block"] == "test").to_numpy()
    fday = ((fl["ts_hour"].to_numpy() - ts0) // 86400).astype(int)
    boot = dict(boot=cfg.bootstrap, seed=seed)
    ft = fl[te]
    ens_m = metrics(y_arr[te], total[te], days=fday[te], f1_key="f1_arr15", **boot)
    hist_keys = ["origin", "how", "month"]
    fl["how"] = cg.hour_of_week(fl["ts_hour"].to_numpy(), fl["o_lon"].to_numpy())
    fl["month"] = pd.DatetimeIndex(pd.to_datetime(fl["ts_dep"], unit="s")).month.to_numpy()
    ft = fl[te]
    base_hist = _hist_mean(fl[f_tr], "arr_delay", hist_keys, ft)
    direct = fit_lgb_flights(f_tr, y_arr).predict(xf[te])
    ens_base = {"hist_mean": metrics(y_arr[te], base_hist, days=fday[te], f1_key="f1_arr15", **boot),
                "lgbm_direct": metrics(y_arr[te], direct, days=fday[te], f1_key="f1_arr15", **boot)}
    ens_vs = {"hist_mean": vs_baseline(y_arr[te], total[te], base_hist, days=fday[te], **boot), "lgbm_direct": vs_baseline(y_arr[te], total[te], direct, days=fday[te], **boot)}
    horizons = {}
    for h in (2, 6, 24):
        m = (ft["horizon_h"] == h).to_numpy()
        if m.any():
            horizons[str(h)] = {"n": int(m.sum()), "mae": float(np.abs(total[te][m] - y_arr[te][m]).mean())}
    per_zone = {}
    for z in sorted(fl["o_zone"].unique()):
        m = te & (fl["o_zone"].to_numpy() == z)
        per_zone[z] = metrics(y_arr[m], total[m], f1_key="f1_arr15", boot=10) if m.sum() >= MIN_ZONE_TEST else "insufficient data"
    lo, hi = conf.interval(total[te])
    ft_month = pd.DatetimeIndex(pd.to_datetime(ft["ts_dep"], unit="s")).strftime("%Y-%m")
    by_month = [{"month": mth, "n": int((ft_month == mth).sum()), "coverage": cf.coverage(y_arr[te][ft_month == mth], lo[ft_month == mth], hi[ft_month == mth])} for mth in sorted(set(ft_month))]
    conformal_card = {"target": TARGET_COVERAGE, "n_cal": int(conf.n_cal), "test_n": int(te.sum()), "test_coverage": cf.coverage(y_arr[te], lo, hi),
                      "test_coverage_by_month": by_month, "mean_width_min": float((hi - lo).mean())}

    ate = (ah["block"] == "test").to_numpy() & (ah["n"] > 0).to_numpy()
    aday = ((ah["ts"].to_numpy() - ts0) // 86400).astype(int)
    ah["how"] = cg.hour_of_week(ah["ts"].to_numpy(), ah["lon"].to_numpy())
    atest = ah[ate]
    n_w = atest["n"].to_numpy(dtype=float)

    def report(y, p, bases, days, w=None) -> dict:
        """metrics + baselines + paired RMSE ratios of one model on one row set (the headline is the RMSE ratio, docs/phase2-design.md, 8)."""
        return {"metrics": metrics(y, p, days=days, w=w, **boot), "baselines": {k: metrics(y, b, days=days, w=w, **boot) for k, b in bases.items()},
                "vs_baseline": {k: vs_baseline(y, p, b, days=days, **boot) for k, b in bases.items()}}

    gcn_card_m, gcn_base, gcn_vs = {}, {}, {}
    for k, (col, head) in enumerate((("y_dep", "y_dep"), ("y_late", "y_late"))):
        yt = atest[col].to_numpy(dtype=float)
        rows_ok = is_tr & (ah["n"] > 0).to_numpy()
        fit, val = split_inner(rows_ok, ah_ts)
        tgt = ah[col].to_numpy(dtype=float)
        nog = wx.LGBModel(cg.GCN_FEATURES, params).fit(xa[fit], tgt[fit], val=(xa[val], tgt[val]), rounds=cfg.rounds)
        bases = {"hist_mean": _hist_mean(ah[is_tr], col, ["airport", "how"], atest, w="n"), "lgbm_no_graph": nog.predict(xa[ate])}
        r = report(yt, pg[atest["s"], atest["a"], k], bases, aday[ate], n_w)
        gcn_card_m[head], gcn_base[head], gcn_vs[head] = r["metrics"], r["baselines"], r["vs_baseline"]
    graph_helps = bool(gcn_vs["y_dep"]["lgbm_no_graph"]["better"])  # headline rule: the whole paired RMSE-ratio interval below 1 (MAE is not used: the GCN trains on Huber, LightGBM on L2)

    nohist_final = fit_cong(is_tr, xa, cg.GCN_FEATURES)
    y_cong = atest["y_nas"].to_numpy(dtype=float)
    cong_bases = {"hist_mean": _hist_mean(ah[is_tr], "y_nas", ["airport", "how"], atest, w="n"), "lgbm_no_hist_prior": nohist_final.predict(xa[ate])}
    cong_rep = report(y_cong, cong_pred[ate], cong_bases, aday[ate], n_w)
    y_wx, y_carr = fl["weather_delay"].to_numpy(dtype=float), (fl["carrier_delay"] + fl["security_delay"]).to_numpy(dtype=float)
    weather_bases = {"mean": np.full(int(te.sum()), y_wx[f_tr].mean()), "origin_mean": _hist_mean(fl[f_tr], "weather_delay", ["origin"], ft)}
    weather_rep = report(y_wx[te], weather_pred[te], weather_bases, fday[te])
    carrier_rep = report(y_carr[te], carrier_pred[te], {"global_mean": np.full(int(te.sum()), y_carr[f_tr].mean())}, fday[te])

    # supplementary: the same comparison on the train out-of-fold predictions (chunks 1..n_folds), where n is about 8x larger than the 10-day test block.
    # Variant selection used these rows (docs/phase2-design.md, 8), so they are evidence of stability, not an unbiased test.
    oof_a = is_tr & (ah["chunk"].to_numpy() >= 1) & (ah["n"].to_numpy() > 0)
    oof_f = f_tr & (fl["chunk"].to_numpy() >= 1)
    hm_oof, const_oof, origin_oof, carr_oof = (np.full(len(x), np.nan) for x in (ah, fl, fl, fl))
    for j in range(1, cfg.n_folds + 1):
        fit_a = is_tr & (ah_ts < chunks.fit_limit(j)) & (ah["n"].to_numpy() > 0)
        sel_a = is_tr & (ah["chunk"].to_numpy() == j)
        hm_oof[sel_a] = _hist_mean(ah[fit_a], "y_nas", ["airport", "how"], ah[sel_a], w="n")
        fit_f = f_tr & (fl_ts < chunks.fit_limit(j))
        sel_f = f_tr & (fl["chunk"].to_numpy() == j)
        const_oof[sel_f], carr_oof[sel_f] = y_wx[fit_f].mean(), y_carr[fit_f].mean()
        origin_oof[sel_f] = _hist_mean(fl[fit_f], "weather_delay", ["origin"], fl[sel_f])
    w_oof = ah.loc[oof_a, "n"].to_numpy(dtype=float)
    cong_rep["oof"] = {"n": int(oof_a.sum()), **report(y_nas[oof_a], cong_pred[oof_a], {"hist_mean": hm_oof[oof_a], "lgbm_no_hist_prior": nohist_pred[oof_a]}, aday[oof_a], w_oof)}
    weather_rep["oof"] = {"n": int(oof_f.sum()), **report(y_wx[oof_f], weather_pred[oof_f], {"mean": const_oof[oof_f], "origin_mean": origin_oof[oof_f]}, fday[oof_f])}
    carrier_rep["oof"] = {"n": int(oof_f.sum()), **report(y_carr[oof_f], carrier_pred[oof_f], {"global_mean": carr_oof[oof_f]}, fday[oof_f])}

    # ---- save artifacts --------------------------------------------------------------------------------------
    gcn_final.save(out / "gcn.npz", feature_names=cg.GCN_FEATURES, nodes=nodes)
    gcn.save_operator(out / "adj.npz", ahat)
    cong_final.save(out / "congestion")
    weather_final.save(out / "weather")
    carrier_final.save(out / "carrier.json")
    sched.save(out / "sched.json")
    hist_final.save(out / "hist.json")
    ens.save(out / "ensemble.json")
    conf.save(out / "conformal.json")
    crs_fit = np.polyfit(fl.loc[f_tr, "distance_km"].to_numpy(dtype=float), fl.loc[f_tr, "crs_elapsed_min"].to_numpy(dtype=float), 1)  # [min per km, fixed min]
    (out / "nodes.json").write_text(json.dumps({"nodes": nodes, "tiers": {a: int(t) for a, t in tiers.items()}, "crs_fit": [float(c) for c in crs_fit],
                                                "train_graph": "mean daily scheduled flights between the graph nodes in the train block"}), encoding="utf-8")

    created = datetime.now(timezone.utc).isoformat(timespec="seconds")
    common = {"trained_on": trained_on, "created_at": created}
    card = {
        "status": "trained", "trained_on": trained_on, "created_at": created, "note": tables.meta.get("note", ""), "source_meta": tables.meta,
        "split": {k: float(v) for k, v in sp.items()} | {"start": float(ts_unique[0]), "end": float(ts_unique[-1] + 3600)},
        "split_dates": {"start": _iso(ts_unique[0]), "train_end": _iso(sp["train_end"]), "calib": [_iso(sp["calib_start"]), _iso(sp["calib_end"])], "test_start": _iso(sp["test_start"]), "end": _iso(ts_unique[-1])},
        "counts": {"nodes": n_nodes, "train_flights": int(f_tr.sum()), "calib_flights": int(cal.sum()), "test_flights": int(te.sum()),
                   "train_airport_hours": int(is_tr.sum()), "test_airport_hours": int(ate.sum())},
        "inference_graph": "world graph from data.routes_graph() (origin 'derived' until OpenFlights/BTS edges are loaded); weights have no node-id features",
        "models": {
            "gcn": {"kind": "ml", **common, "name": "2-layer GCN (numpy), heads: mean departure delay, late-aircraft delay", "hidden": cfg.gcn_hidden, "features": f_count,
                    "metrics": gcn_card_m, "baselines": gcn_base, "vs_baseline": gcn_vs, "graph_helps": graph_helps,
                    "graph_helps_rule": "departure-delay RMSE ratio vs the same features without the graph, whole 95% paired block-bootstrap interval below 1"},
            "congestion": {"kind": "ml", **common, "name": "LightGBM airport-hour NAS-type delay -> stress index", **cong_rep,
                           "hist_prior": {"targets": list(cg.HistPrior.TARGETS), "shrink": hist_final.shrink, "airports": len(hist_final.airports),
                                          "note": "train-only mean by airport and airport x hour-of-week; leave-one-week-out inside the fit window"}},
            "weather": {"kind": "ml", **common, "name": "LightGBM flight-level weather delay", **weather_rep},
            "routing": {"kind": "algorithmic", "trained_on": "n/a", "name": "great-circle + range-aware stops + jet-wind time", "label": routing.LABEL},
            "bird": {"kind": "prior", "trained_on": "n/a", "name": "ecology prior (season, local solar time, latitude, flyway), optional eBird blend", "label": bird_mod.LABEL},
            "carrier": {"kind": "lookup", **common, "name": "mean carrier+security delay by airline x origin x month", **carrier_rep},
            "ensemble": {"kind": "stacked", **common, "name": "non-negative linear stacker per weather zone + split-conformal 80% intervals",
                         "metrics": ens_m, "baselines": ens_base, "vs_baseline": ens_vs, "horizons": horizons, "per_zone": per_zone, "conformal": conformal_card,
                         "weights_pooled": ens.pooled, "weights_by_zone": ens.zones},
        },
    }
    (out / "model_card.json").write_text(json.dumps(card, indent=2, default=float), encoding="utf-8")
    return card


def main(argv=None) -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--source", choices=["synthetic", "real"], required=True)
    ap.add_argument("--out", help="artifact directory (synthetic default: a temp dir, never backend/models)")
    ap.add_argument("--quick", action="store_true")
    ap.add_argument("--force", action="store_true", help="allow synthetic output to replace a trained_on=real card")
    ap.add_argument("--days", type=int, default=45, help="synthetic: days to simulate")
    ap.add_argument("--airports", type=int, default=None, help="synthetic: number of US airports")
    ap.add_argument("--months", type=int, default=30, help="real: number of latest complete BTS months")
    ap.add_argument("--work", help="real: download/cache directory")
    ap.add_argument("--threads", type=int, default=1, help="LightGBM threads (1 is fastest for small data; use more for real runs)")
    args = ap.parse_args(argv)
    out = Path(args.out) if args.out else Path(tempfile.mkdtemp(prefix="aerogrid-models-"))
    if args.source == "synthetic":
        from . import synth

        tables = synth.make_synthetic(n_days=args.days, n_airports=args.airports)
    else:
        from . import real

        tables = real.load_real(months=args.months, work_dir=Path(args.work) if args.work else out / "work", log=lambda *a: print(*a, file=sys.stderr, flush=True))
    card = train(tables, out, quick=args.quick, force=args.force, config=replace(CONFIGS['quick' if args.quick else 'full'], threads=args.threads))
    print(f"wrote {out} (trained_on={card['trained_on']}, ensemble MAE {card['models']['ensemble']['metrics']['mae']:.2f} min, n={card['models']['ensemble']['metrics']['n']})")


if __name__ == "__main__":
    main()
