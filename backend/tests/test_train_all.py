import json
import math

import numpy as np
import pytest

from aerogrid.ml import conformal as cf
from aerogrid.ml import ensemble as en
from aerogrid.ml import gcn
from aerogrid.train import synth, train_all

ARTIFACTS = ["gcn.npz", "adj.npz", "congestion", "weather", "carrier.json", "sched.json", "hist.json", "ensemble.json", "conformal.json", "nodes.json", "model_card.json"]


@pytest.fixture(scope="module")
def trained(tmp_path_factory):
    out = tmp_path_factory.mktemp("models")
    tables = synth.make_synthetic(n_days=20, n_airports=7, seed=5, flights_scale=0.8)
    card = train_all.train(tables, out, quick=True)
    return out, card, tables


def test_all_artifacts_are_written(trained):
    out, _, _ = trained
    for name in ARTIFACTS:
        assert (out / name).exists(), name
    meta = json.loads((out / "nodes.json").read_text())
    assert meta["nodes"] and len(meta["crs_fit"]) == 2 and meta["crs_fit"][0] > 0, "scheduled block time grows with distance"


def test_model_card_labels_every_model_with_kind_and_trained_on(trained):
    _, card, _ = trained
    assert card["trained_on"] == "synthetic" and "SYNTHETIC" in card["note"].upper()
    kinds = {k: v["kind"] for k, v in card["models"].items()}
    assert kinds == {"gcn": "ml", "congestion": "ml", "weather": "ml", "routing": "algorithmic", "bird": "prior", "carrier": "lookup", "ensemble": "stacked"}
    assert all(v["trained_on"] in {"synthetic", "n/a"} for v in card["models"].values())
    assert card["models"]["routing"]["trained_on"] == "n/a" and card["models"]["bird"]["label"] == "prior-based estimate"
    assert card["models"]["gcn"]["trained_on"] == "synthetic"
    assert card["status"] == "trained"


def test_card_reports_window_split_and_counts(trained):
    _, card, tables = trained
    s = card["split"]
    assert s["train_end"] < s["calib_start"] < s["calib_end"] < s["test_start"]
    assert card["counts"]["train_flights"] > card["counts"]["calib_flights"] > 0 and card["counts"]["test_flights"] > 0
    assert card["counts"]["nodes"] == 7


def test_metrics_exist_with_n_baselines_and_bootstrap_ci(trained):
    _, card, _ = trained
    g = card["models"]["gcn"]
    for head in ("y_dep", "y_late"):
        m = g["metrics"][head]
        assert m["n"] > 0 and all(math.isfinite(m[k]) for k in ("mae", "rmse"))
        assert m["mae_ci95"][0] <= m["mae"] <= m["mae_ci95"][1]
    assert {"hist_mean", "lgbm_no_graph"} <= set(g["baselines"]["y_dep"])
    assert isinstance(g["graph_helps"], bool)
    e = card["models"]["ensemble"]
    assert {"mae", "rmse", "f1_arr15", "n"} <= set(e["metrics"]) and "hist_mean" in e["baselines"]
    assert e["horizons"].keys() >= {"2", "24"}


def test_vs_baseline_counts_a_win_only_when_the_whole_interval_is_below_one():
    rng = np.random.default_rng(0)
    days = np.repeat(np.arange(40), 100)
    y = rng.normal(0, 1, len(days))
    good, zero = y + rng.normal(0, 0.1, len(days)), np.zeros(len(days))
    win = train_all.vs_baseline(y, good, zero, days=days, boot=200)
    assert win["better"] and win["ci95"][1] < 1 and win["n_days"] == 40
    tie = train_all.vs_baseline(y, zero, zero, days=days, boot=200)
    assert not tie["better"] and tie["rmse_ratio"] == pytest.approx(1.0) and tie["ci95"] == pytest.approx([1.0, 1.0])


def test_card_has_paired_ratios_extra_baselines_and_out_of_fold_evidence(trained):
    _, card, _ = trained
    m = card["models"]
    assert {"hist_mean", "lgbm_no_hist_prior"} <= set(m["congestion"]["baselines"]) and {"mean", "origin_mean"} <= set(m["weather"]["baselines"])
    for name in ("congestion", "weather", "carrier"):
        for part in (m[name], m[name]["oof"]):
            assert set(part["vs_baseline"]) == set(part["baselines"]) and part["metrics"]["n"] > 0
            for v in part["vs_baseline"].values():
                assert 0 < v["ci95"][0] <= v["ci95"][1] and v["better"] == (v["ci95"][1] < 1) and v["n_days"] > 0
        assert m[name]["oof"]["n"] > m[name]["metrics"]["n"] * 0.5, "the out-of-fold rows are the large sample"
    assert {"y_dep", "y_late"} == set(m["gcn"]["vs_baseline"]) and {"hist_mean", "lgbm_direct"} == set(m["ensemble"]["vs_baseline"])
    assert m["congestion"]["hist_prior"]["shrink"] == 60.0 and "rmse_weighted" in m["congestion"]["metrics"]


def test_gcn_artifact_keeps_the_feature_list_without_the_prior(trained):
    out, _, _ = trained
    from aerogrid.ml import congestion as cg

    assert gcn.GCN.load(out / "gcn.npz")[1]["feature_names"] == cg.GCN_FEATURES
    assert en.Bundle(out).congestion.gbm.feature_names == cg.AIRPORT_FEATURES


def test_per_zone_reports_insufficient_data_instead_of_numbers_for_thin_zones(trained):
    _, card, _ = trained
    pz = card["models"]["ensemble"]["per_zone"]
    assert pz and all((v == "insufficient data") or ("mae" in v and v["n"] >= 1000) for v in pz.values())


def test_ensemble_has_no_bird_effect_because_the_synthetic_truth_has_none(trained):
    out, _, tables = trained
    ens = en.Ensemble.load(out / "ensemble.json")
    w = ens.pooled["weights"]
    assert w["weather"] > 0.2 and w["carrier"] > 0.2
    assert ens.pooled["baseline"] >= 0 and all(v >= 0 for v in w.values())
    assert w["bird"] * 0.8 < 1.0, "bird may not explain more than a minute on average when it has no effect"


def test_conformal_coverage_is_reported_and_roughly_nominal(trained):
    out, card, _ = trained
    cov = card["models"]["ensemble"]["conformal"]
    assert cov["target"] == 0.8 and cov["n_cal"] > 100
    assert 0.6 <= cov["test_coverage"] <= 0.97 and cov["test_n"] > 100
    assert cf.SplitConformal.load(out / "conformal.json").alpha == pytest.approx(0.2)


def test_saved_gcn_runs_on_a_bigger_graph(trained):
    out, _, _ = trained
    model, meta = gcn.GCN.load(out / "gcn.npz")
    n = 25
    ahat = gcn.operator_symmetric(__import__("scipy.sparse", fromlist=["x"]).random(n, n, density=0.2, random_state=0))
    x = np.random.default_rng(0).normal(size=(2, n, len(meta["feature_names"])))
    assert model.predict(ahat, x).shape == (2, n, 2)


def test_a_window_too_short_for_all_blocks_is_rejected_with_a_clear_message(tmp_path):
    tables = synth.make_synthetic(n_days=8, n_airports=4, seed=1, flights_scale=0.5)
    with pytest.raises(ValueError, match="test"):
        train_all.train(tables, tmp_path, quick=True)


def test_synthetic_run_refuses_to_overwrite_a_real_card(tmp_path):
    (tmp_path / "model_card.json").write_text(json.dumps({"trained_on": "real", "status": "trained"}))
    tables = synth.make_synthetic(n_days=14, n_airports=4, seed=1, flights_scale=0.5)
    with pytest.raises(RuntimeError, match="real"):
        train_all.train(tables, tmp_path, quick=True)
    train_all.train(tables, tmp_path, quick=True, force=True)
    assert json.loads((tmp_path / "model_card.json").read_text())["trained_on"] == "synthetic"
