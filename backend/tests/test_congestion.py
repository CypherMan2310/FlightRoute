import numpy as np
import pandas as pd
import pytest

from aerogrid.ml import congestion as cg
from aerogrid.ml import weather as wx

T0 = pd.Timestamp("2026-03-02 00:00", tz="UTC").timestamp()  # a Monday
LON = {"AAA": 0.0, "BBB": 0.0}


def flights(n_weeks=2):
    """AAA: 3 departures every hour on Mondays only (local = UTC at lon 0); BBB: 1 arrival per hour on Mondays."""
    rows = []
    for w in range(n_weeks):
        for h in range(24):
            t = T0 + w * 7 * 86400 + h * 3600 + 600
            rows += [{"origin": "AAA", "dest": "BBB", "ts_dep": t, "ts_arr": t + 3600}] * 3
    return pd.DataFrame(rows)


def test_sched_profile_counts_mean_weekly_flights_per_hour_of_week():
    prof = cg.SchedProfile().fit(flights(2), LON, {"AAA": 1, "BBB": 2})
    assert prof.deps["AAA"][0] == pytest.approx(3.0) and prof.deps["AAA"][24] == 0  # Monday 00:00 vs Tuesday 00:00
    assert prof.arrs["BBB"][1] == pytest.approx(3.0)  # arrivals land one hour later


def test_sched_profile_lookup_and_tier_fallback_for_unknown_airports():
    prof = cg.SchedProfile().fit(flights(2), LON, {"AAA": 1, "BBB": 2})
    deps, arrs = prof.lookup(["AAA", "ZZZ"], [1, 1], [T0 + 3600, T0 + 3600], [0.0, 0.0])
    assert deps[0] == pytest.approx(3.0)
    assert deps[1] == pytest.approx(3.0)  # ZZZ is tier 1: falls back to the tier-1 mean profile (only AAA)
    d2, _ = prof.lookup(["QQQ"], [3], [T0], [0.0])
    assert d2[0] == 0.0  # no tier-3 airport in the fit: zero, never NaN


def test_sched_profile_roundtrip(tmp_path):
    prof = cg.SchedProfile().fit(flights(2), LON, {"AAA": 1, "BBB": 2})
    prof.save(tmp_path / "sched.json")
    again = cg.SchedProfile.load(tmp_path / "sched.json")
    assert again.lookup(["AAA"], [1], [T0 + 7200], [0.0])[0][0] == pytest.approx(3.0)


def hist_rows(n_weeks=4, spike=None):
    """Airport-hour rows (AAA tier 1, BBB and CCC tier 2) for n_weeks full weeks starting on a Monday, 10 flights per hour, lon 0.
    A flat NAS delay of 10 / 2 / 4 minutes; `spike` = (airport, week, hour_of_week) adds 1000 minutes to that one row."""
    rows = []
    for a, tier, level in (("AAA", 1, 10.0), ("BBB", 2, 2.0), ("CCC", 2, 4.0)):
        for w in range(n_weeks):
            for h in range(168):
                y = level + (1000.0 if spike == (a, w, h) else 0.0)
                rows.append({"airport": a, "tier": tier, "ts": T0 + w * 7 * 86400 + h * 3600.0, "week": w, "n": 10, "y_nas": y, "y_dep": y, "y_late": y})
    df = pd.DataFrame(rows)
    df["how"] = cg.hour_of_week(df["ts"], 0.0)
    return df


def fit_prior(df, folds=True):
    return cg.HistPrior().fit(df["airport"], df["tier"], df["how"], df["n"], {t: df[t] for t in cg.HistPrior.TARGETS}, folds=df["week"] if folds else None)


def test_hist_prior_follows_each_airport_and_falls_back_to_the_tier_then_global():
    df = hist_rows()
    prior = fit_prior(df, folds=False)
    ts = [T0 + 3600] * 4
    got = prior.lookup(["AAA", "BBB", "ZZZ", "QQQ"], [1, 2, 2, 3], ts, [0.0] * 4)
    assert list(got) == [f"hp_{t[2:]}_{lvl}" for t in cg.HistPrior.TARGETS for lvl in ("ap", "how")] == cg.HIST_FEATURES
    assert got["hp_nas_how"][0] == pytest.approx(10.0, abs=0.2) and got["hp_nas_how"][1] == pytest.approx(2.0, abs=0.1)
    assert 2.0 < got["hp_nas_ap"][2] < 4.0 and got["hp_nas_how"][2] == pytest.approx(got["hp_nas_ap"][2], abs=0.05), "an unseen tier-2 airport gets the tier-2 level"
    glob = (10 + 2 + 4) / 3
    assert got["hp_nas_ap"][3] == pytest.approx(glob, abs=0.3), "an unseen tier falls back to the global mean"


def test_hist_prior_is_shrunk_toward_its_parent_when_a_cell_is_thin():
    df = hist_rows(n_weeks=1, spike=("BBB", 0, 5))  # one week: the spiked cell holds 10 flights
    prior = fit_prior(df, folds=False)
    cell = prior.lookup(["BBB"], [2], [T0 + 5 * 3600], [0.0])["hp_nas_how"][0]
    assert 2.0 < cell < 500.0, "a thin cell (10 flights, shrink 60) is pulled well below the raw 1002 of its single observation"


def test_out_of_fold_value_never_sees_its_own_week():
    df = hist_rows(spike=("AAA", 0, 5))
    prior = fit_prior(df)
    row = df[(df["airport"] == "AAA") & (df["week"] == 0) & (df["how"] == 5)]
    oof = prior.oof_lookup(row["airport"], row["tier"], row["ts"], [0.0], row["week"])
    in_sample = prior.lookup(row["airport"], row["tier"], row["ts"], [0.0])
    without = fit_prior(df[df["week"] != 0], folds=False).lookup(row["airport"], row["tier"], row["ts"], [0.0])
    for c in cg.HIST_FEATURES:
        assert oof[c][0] == pytest.approx(without[c][0]), f"{c}: leave-one-week-out equals a table fitted without that week"
    assert oof["hp_nas_how"][0] < 15.0 < in_sample["hp_nas_how"][0], "the plain table has absorbed the spike, the out-of-fold value has not"
    with pytest.raises(ValueError, match="folds"):
        fit_prior(df, folds=False).oof_lookup(row["airport"], row["tier"], row["ts"], [0.0], row["week"])


def test_hist_prior_roundtrip(tmp_path):
    df = hist_rows(n_weeks=2)
    prior = fit_prior(df, folds=False)
    prior.save(tmp_path / "hist.json")
    again = cg.HistPrior.load(tmp_path / "hist.json")
    args = (["AAA", "CCC", "ZZZ"], [1, 2, 3], [T0, T0 + 7200, T0 + 9 * 3600], [0.0] * 3)
    a, b = prior.lookup(*args), again.lookup(*args)
    assert all(np.allclose(a[c], b[c]) for c in cg.HIST_FEATURES)


def test_gcn_features_are_the_airport_features_without_the_prior():
    assert cg.AIRPORT_FEATURES[:2] == ["sched_deps", "sched_arrs"] and set(cg.AIRPORT_FEATURES) - set(cg.GCN_FEATURES) == set(cg.HIST_FEATURES)


def raw_airport_frame(n, seed=0):
    rng = np.random.default_rng(seed)
    raw = pd.DataFrame({
        "airport": rng.choice(["AAA", "BBB", "CCC"], n), "tier": 2, "lon": -80.0, "zone": "temperate",
        "ts": T0 + rng.integers(0, 120 * 24, n) * 3600.0, "horizon_h": rng.choice([2, 6, 24], n),
        "sched_deps": rng.uniform(0, 40, n), "sched_arrs": rng.uniform(0, 40, n), **{c: 0.0 for c in cg.HIST_FEATURES},
    })
    for k, v in {**wx.NEUTRAL_OBS, **wx.NEUTRAL_FC}.items():
        raw[k] = v
    raw["obs_tstorm"] = (rng.random(n) < 0.15).astype(float)
    return raw


def test_airport_features_have_exactly_the_model_columns():
    x = cg.build_airport_features(raw_airport_frame(5))
    assert list(x.columns) == cg.AIRPORT_FEATURES and not x.isna().any().any()
    assert x["tier"].eq(2).all() and x["zone_temperate"].eq(1).all()


def test_airport_features_reject_missing_raw_column():
    with pytest.raises(KeyError):
        cg.build_airport_features(raw_airport_frame(5).drop(columns=["sched_arrs"]))


def test_congestion_model_learns_load_and_storms_and_maps_stress(tmp_path):
    raw = raw_airport_frame(3000)
    x = cg.build_airport_features(raw)
    rng = np.random.default_rng(1)
    y = 0.4 * raw["sched_deps"] * (1 + 2 * raw["obs_tstorm"]) + rng.normal(0, 1, len(raw))
    n = rng.integers(1, 40, len(raw))
    tr, te = slice(0, 2400), slice(2400, 3000)
    model = cg.CongestionModel().fit(x.iloc[tr], y.iloc[tr].to_numpy(), n[tr], rounds=150)
    pred = model.predict(x.iloc[te])
    assert np.corrcoef(pred, y.iloc[te])[0, 1] > 0.9
    s = model.stress(x.iloc[te])
    assert s.min() >= 0 and s.max() <= 100
    order = np.argsort(pred)
    assert np.all(np.diff(s[order]) >= -1e-9)  # stress is monotone in the predicted delay
    model.save(tmp_path / "cong")
    again = cg.CongestionModel.load(tmp_path / "cong")
    assert np.allclose(again.predict(x.iloc[te]), pred) and np.allclose(again.stress(x.iloc[te]), s)


def test_zero_flight_rows_are_ignored_in_training():
    raw = raw_airport_frame(600)
    x = cg.build_airport_features(raw)
    y = np.full(len(raw), 5.0)
    y[:100] = 1e6  # garbage targets on rows with no flights
    n = np.ones(len(raw), dtype=int)
    n[:100] = 0
    model = cg.CongestionModel().fit(x, y, n, rounds=30)
    assert model.predict(x).max() < 50
