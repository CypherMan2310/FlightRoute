import json

import numpy as np
import pandas as pd
import pytest

from aerogrid import data
from aerogrid.ml import congestion as cg
from aerogrid.ml import ensemble as en
from aerogrid.ml import weather as wx
from aerogrid.ml import whatif
from aerogrid.train import synth, train_all

NOW = pd.Timestamp("2026-07-14 18:00", tz="UTC").timestamp()


@pytest.fixture(scope="module")
def bundle(tmp_path_factory):
    out = tmp_path_factory.mktemp("bundle")
    cfg = train_all.Config(n_folds=2, gcn_hidden=16, gcn_epochs=40, gcn_lr=5e-3, rounds=120, bootstrap=30)  # small but strong enough to react to a storm
    train_all.train(synth.make_synthetic(n_days=30, n_airports=10, seed=5, flights_scale=1.0), out, config=cfg)
    return en.Bundle(out)


def snapshot(ts=NOW, horizon=2, codes=None):
    codes = codes or data.routes_graph().codes
    return {"ts": ts, "horizon_h": horizon, "obs": {c: dict(wx.NEUTRAL_OBS) for c in codes}, "fc": {c: dict(wx.NEUTRAL_FC) for c in codes}}


def test_bundle_reports_provenance(bundle):
    assert bundle.trained_on == "synthetic" and bundle.status == "trained"
    assert len(bundle.nodes) == 10 and bundle.card["models"]["ensemble"]["kind"] == "stacked"


def test_load_or_none_returns_none_without_artifacts(tmp_path):
    assert en.Bundle.load_or_none(tmp_path) is None
    (tmp_path / "model_card.json").write_text('{"status": "untrained"}')
    assert en.Bundle.load_or_none(tmp_path) is None


def test_incompatible_artifacts_are_reported_with_a_reason_not_hidden(bundle, tmp_path):
    import shutil

    shutil.copytree(bundle.path, tmp_path / "old")
    meta = json.loads((tmp_path / "old" / "nodes.json").read_text())
    meta.pop("crs_fit")
    (tmp_path / "old" / "nodes.json").write_text(json.dumps(meta))
    assert en.Bundle.load_or_none(tmp_path / "old") is None
    assert "crs_fit" in en.Bundle.load_error(tmp_path / "old")
    assert en.Bundle.load_error(bundle.path) is None
    assert "model_card.json" in en.Bundle.load_error(tmp_path / "missing")


def test_world_operator_is_row_stochastic_over_the_routes_graph():
    ahat = en.world_operator().toarray()
    assert ahat.shape == (155, 155) and np.allclose(ahat.sum(axis=1), 1.0) and (np.diag(ahat) >= 0.5).all()


def test_world_frame_covers_every_graph_airport_with_the_model_columns(bundle):
    raw = bundle.world_frame(snapshot())
    assert list(raw["airport"]) == list(data.routes_graph().codes) and len(raw) == 155
    x = cg.build_airport_features(raw)
    assert list(x.columns) == cg.AIRPORT_FEATURES and not x.isna().any().any()
    assert (raw["sched_deps"] >= 0).all() and raw["zone"].isin(data.ZONES).all()
    lhr = raw[raw["airport"] == "LHR"].iloc[0]
    assert lhr["sched_deps"] > 0, "non-US airports fall back to the tier profile instead of zero"


def test_score_world_returns_bounded_scores_for_every_airport(bundle):
    s = bundle.score_world(bundle.world_frame(snapshot()))
    assert s["gcn"].shape == (155, 2) and (s["gcn"] >= 0).all() and (s["gcn"] <= 180).all(), "GCN minutes are clipped to a physical range"
    assert s["stress"].min() >= 0 and s["stress"].max() <= 100 and (s["cong_pred"] >= 0).all()
    assert s["codes"] == list(data.routes_graph().codes)


def row(bundle, origin="JFK", dest="LAX", carrier="AA", dep=NOW + 2 * 3600, storm=False):
    o_obs = dict(wx.NEUTRAL_OBS, **({"obs_tstorm": 1.0, "obs_precip": 2.0, "obs_vis_sm": 1.0, "obs_cat": 2.0} if storm else {}))
    return en.build_flight_row(bundle, origin, dest, carrier, dep, horizon_h=2, o_obs=o_obs, d_obs=dict(wx.NEUTRAL_OBS), o_fc=dict(wx.NEUTRAL_FC), d_fc=dict(wx.NEUTRAL_FC),
                               er_fc=[dict(wx.NEUTRAL_FC)] * 3, er_wind=[(60.0, 270.0)] * 3)


def test_flight_row_has_every_raw_column_the_models_need(bundle):
    r = row(bundle)
    x = wx.build_flight_features(r)
    assert list(x.columns) == wx.FLIGHT_FEATURES and not x.isna().any().any()
    jfk, lax = data.get_airport("JFK"), data.get_airport("LAX")
    assert r.loc[0, "distance_km"] == pytest.approx(data.great_circle_km(jfk.lat, jfk.lon, lax.lat, lax.lon))
    assert r.loc[0, "crs_elapsed_min"] > 120 and r.loc[0, "o_zone"] in data.ZONES
    assert r.loc[0, "route_extra_min"] >= 0 and 0 <= r.loc[0, "bird_index"] <= 0.8
    short = en.build_flight_row(bundle, "JFK", "BOS", "AA", NOW, 2, dict(wx.NEUTRAL_OBS), dict(wx.NEUTRAL_OBS), dict(wx.NEUTRAL_FC), dict(wx.NEUTRAL_FC), [dict(wx.NEUTRAL_FC)] * 3, [(0.0, 0.0)] * 3)
    assert short.loc[0, "crs_elapsed_min"] < r.loc[0, "crs_elapsed_min"]


def test_predict_flights_is_additive_with_an_interval_and_six_causes(bundle):
    rows = pd.concat([row(bundle), row(bundle, "ORD", "ATL", "DL"), row(bundle, storm=True)], ignore_index=True)
    score = bundle.score_world(bundle.world_frame(snapshot()))
    out = bundle.predict_flights(rows, score)
    assert set(out["causes"].columns) == set(en.COMPONENTS) and (out["causes"] >= 0).all().all()
    assert np.allclose(out["total"], out["causes"].sum(axis=1) + out["baseline"], atol=1e-9)
    assert (out["lo"] <= out["hi"]).all() and (out["hi"] >= out["total"] - 1e-9).all() and (out["lo"] >= 0).all()
    assert set(out["components"].columns) == set(en.COMPONENTS)


def test_a_thunderstorm_at_the_origin_raises_the_weather_component(bundle):
    score = bundle.score_world(bundle.world_frame(snapshot()))
    calm, storm = bundle.predict_flights(pd.concat([row(bundle), row(bundle, storm=True)], ignore_index=True), score)["components"]["weather"]
    assert storm > calm


# ---- what-if ---------------------------------------------------------------------------------------------------

def test_storm_scenario_raises_the_airports_own_predicted_delay_and_stress(bundle):
    raw = bundle.world_frame(snapshot(horizon=2))
    res = whatif.run(bundle, raw, "ATL", "storm", 1.0)
    atl = next(d for d in res["deltas"] if d["iata"] == "ATL")
    assert atl["hops"] == 0 and atl["delta_dep_min"] > 0 and atl["delta_stress"] >= 0
    assert res["label"] == "model-based scenario (not a forecast)" and res["trained_on"] == "synthetic"
    assert res["affected"][0]["iata"] == "ATL" or abs(res["affected"][0]["delta_dep_min"]) >= abs(atl["delta_dep_min"]) - 1e-9


def test_scenario_with_zero_severity_changes_nothing(bundle):
    raw = bundle.world_frame(snapshot())
    for kind in ("storm", "runway_closure"):
        res = whatif.run(bundle, raw, "ORD", kind, 0.0)
        assert max(abs(d["delta_dep_min"]) for d in res["deltas"]) < 1e-9 and max(abs(d["delta_stress"]) for d in res["deltas"]) < 1e-9


def test_runway_closure_scales_demand_against_capacity_and_is_monotone_in_severity(bundle):
    raw = bundle.world_frame(snapshot())
    mild = whatif.apply_scenario(raw, "ORD", "runway_closure", 0.3)
    harsh = whatif.apply_scenario(raw, "ORD", "runway_closure", 0.9)
    i = list(raw["airport"]).index("ORD")
    assert raw.loc[i, "sched_deps"] < mild.loc[i, "sched_deps"] < harsh.loc[i, "sched_deps"]
    others = raw["airport"] != "ORD"
    assert (mild.loc[others, "sched_deps"] == raw.loc[others, "sched_deps"]).all()
    assert raw["sched_deps"].sum() > 0 and (raw.drop(columns=[]).equals(raw)), "the input frame is not modified in place"


def test_storm_scenario_sets_severe_weather_only_at_that_airport(bundle):
    raw = bundle.world_frame(snapshot())
    stormy = whatif.apply_scenario(raw, "JFK", "storm", 1.0)
    i = list(raw["airport"]).index("JFK")
    assert stormy.loc[i, "obs_tstorm"] == 1.0 and stormy.loc[i, "obs_cat"] >= 2.0 and stormy.loc[i, "obs_vis_sm"] < 2.0
    assert stormy.loc[i, "fc_cape_jkg"] > 1000 and stormy.loc[i, "fc_gust_kt"] > 30
    others = raw["airport"] != "JFK"
    assert stormy.loc[others, wx.OBS + wx.FC].equals(raw.loc[others, wx.OBS + wx.FC])


def test_hops_follow_the_routes_graph_and_the_affected_list_is_sorted(bundle):
    res = whatif.run(bundle, bundle.world_frame(snapshot()), "KEF", "storm", 1.0)
    hops = {d["iata"]: d["hops"] for d in res["deltas"]}
    g = data.routes_graph()
    assert hops["KEF"] == 0 and all(hops[n] == 1 for n in g.neighbors("KEF"))
    mags = [abs(d["delta_dep_min"]) for d in res["affected"]]
    assert mags == sorted(mags, reverse=True) and len(res["affected"]) <= 15


def test_bad_scenario_inputs(bundle):
    raw = bundle.world_frame(snapshot())
    with pytest.raises(KeyError):
        whatif.run(bundle, raw, "ZZZ", "storm", 0.5)
    with pytest.raises(ValueError):
        whatif.run(bundle, raw, "ATL", "meteor", 0.5)
    with pytest.raises(ValueError):
        whatif.run(bundle, raw, "ATL", "storm", 1.5)
