import numpy as np
import pandas as pd
import pytest

from aerogrid.ml import congestion as cg
from aerogrid.ml import weather as wx
from aerogrid.train import synth

CAUSES = ["carrier_delay", "weather_delay", "nas_delay", "security_delay", "late_aircraft_delay"]


@pytest.fixture(scope="module")
def tables():
    return synth.make_synthetic(n_days=16, n_airports=8, seed=3, flights_scale=0.6)


def test_metadata_labels_the_data_synthetic(tables):
    assert tables.meta["trained_on"] == "synthetic"
    assert "SYNTHETIC" in tables.meta["note"].upper()
    assert len(tables.nodes) == 8 and set(tables.airports["iata"]) == set(tables.nodes)


def test_same_seed_same_tables_and_different_seed_differs():
    a = synth.make_synthetic(n_days=6, n_airports=5, seed=1, flights_scale=0.5)
    b = synth.make_synthetic(n_days=6, n_airports=5, seed=1, flights_scale=0.5)
    c = synth.make_synthetic(n_days=6, n_airports=5, seed=2, flights_scale=0.5)
    pd.testing.assert_frame_equal(a.flights, b.flights)
    pd.testing.assert_frame_equal(a.airport_hour, b.airport_hour)
    assert not a.flights["arr_delay"].equals(c.flights["arr_delay"])


def test_airport_hour_has_every_airport_and_hour_once_with_one_horizon_per_snapshot(tables):
    ah = tables.airport_hour
    assert len(ah) == 8 * 16 * 24 and not ah.duplicated(["airport", "ts"]).any()
    assert ah.groupby("ts")["horizon_h"].nunique().eq(1).all()
    assert set(ah["horizon_h"]) <= {2, 6, 24} and len(set(ah["horizon_h"])) == 3
    needed = {"airport", "ts", "horizon_h", "tier", "lon", "zone", "y_dep", "y_late", "y_nas", "n"} | set(wx.OBS + wx.FC)
    assert needed <= set(ah.columns)
    assert not ah[list(needed - {"y_dep", "y_late", "y_nas"})].isna().any().any()


def test_flights_have_every_raw_column_the_models_need(tables):
    f = tables.flights
    raw = {"ts_dep", "horizon_h", "o_lon", "o_tier", "d_tier", "o_zone", "distance_km", "crs_elapsed_min"} | set(wx.ENROUTE)
    raw |= {f"{s}_{c}" for s in "od" for c in wx.OBS + wx.FC}
    assert raw <= set(f.columns)
    x = wx.build_flight_features(f)
    assert list(x.columns) == wx.FLIGHT_FEATURES and not x.isna().any().any()
    assert {"origin", "dest", "carrier", "ts_arr", "ts_hour", "route_extra_min", "bird_index", "arr_delay", "dep_delay"} | set(CAUSES) <= set(f.columns)
    assert (f["origin"] != f["dest"]).all() and (f["ts_arr"] > f["ts_dep"]).all()


def test_bts_conventions_for_cause_columns(tables):
    f = tables.flights
    late = f["arr_delay"] >= 15
    assert (f.loc[~late, CAUSES] == 0).all().all()  # causes are 0-filled when ArrDel15 = 0
    assert np.allclose(f.loc[late, CAUSES].sum(axis=1), f.loc[late, "arr_delay"], atol=1.0)
    assert (f["arr_delay"] >= 0).all() and 0.05 < late.mean() < 0.5


def test_every_flight_joins_an_airport_hour_row_with_the_same_horizon(tables):
    key = tables.airport_hour.set_index(["airport", "ts"])
    joined = key.reindex(pd.MultiIndex.from_frame(tables.flights[["origin", "ts_hour"]]))
    assert not joined["horizon_h"].isna().any()
    assert (joined["horizon_h"].to_numpy() == tables.flights["horizon_h"].to_numpy()).all()


def test_airport_hour_targets_are_flight_means_and_n_counts_flights(tables):
    f, ah = tables.flights, tables.airport_hour
    g = f.groupby(["origin", "ts_hour"]).agg(n=("arr_delay", "size"), dep=("dep_delay", "mean"), late=("late_aircraft_delay", "mean"), nas=("nas_delay", "mean"))
    row = ah.set_index(["airport", "ts"]).loc[g.index]
    assert (row["n"].to_numpy() == g["n"].to_numpy()).all()
    assert np.allclose(row["y_dep"], g["dep"]) and np.allclose(row["y_nas"], g["nas"]) and np.allclose(row["y_late"], g["late"])
    empty = ah[ah["n"] == 0]
    assert len(empty) > 0 and empty[["y_dep", "y_late", "y_nas"]].isna().all().all()


def test_weather_signal_exists_and_bird_has_no_effect(tables):
    f = tables.flights
    assert f.loc[f["o_obs_tstorm"] == 1, "weather_delay"].mean() > 1.5 * f.loc[f["o_obs_tstorm"] == 0, "weather_delay"].mean() > 0
    assert np.corrcoef(f["o_fc_gust_kt"], f["weather_delay"])[0, 1] > 0.15
    assert abs(np.corrcoef(f["bird_index"], f["arr_delay"])[0, 1]) < 0.1  # synthetic truth has no bird effect, so no model may find one


def test_split_boundaries_are_ordered_with_gaps(tables):
    s, ts = tables.split, tables.airport_hour["ts"]
    assert ts.min() <= s["train_end"] < s["calib_start"] < s["calib_end"] < s["test_start"] <= ts.max() + 3600
    assert s["calib_start"] - s["train_end"] >= 86400 and s["test_start"] - s["calib_end"] >= 86400


def test_airport_features_can_be_built_after_attaching_the_sched_profile(tables):
    prof = cg.SchedProfile().fit(tables.flights, dict(zip(tables.airports["iata"], tables.airports["lon"])), dict(zip(tables.airports["iata"], tables.airports["tier"])))
    ah = tables.airport_hour.copy()
    ah["sched_deps"], ah["sched_arrs"] = prof.lookup(ah["airport"], ah["tier"], ah["ts"], ah["lon"])
    ah[cg.HIST_FEATURES] = 0.0
    x = cg.build_airport_features(ah)
    assert list(x.columns) == cg.AIRPORT_FEATURES and not x.isna().any().any()
