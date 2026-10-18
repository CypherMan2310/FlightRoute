import numpy as np
import pandas as pd
import pytest
from scipy import sparse

from aerogrid.ml import explain as ex
from aerogrid.ml import gcn
from aerogrid.ml import weather as wx


def storm_model():
    rng = np.random.default_rng(0)
    n = 1500
    X = pd.DataFrame(0.0, index=range(n), columns=wx.FLIGHT_FEATURES)
    X["o_obs_tstorm"] = (rng.random(n) < 0.3).astype(float)
    X["er_max_gust_kt"] = rng.uniform(5, 50, n)
    y = 20 * X["o_obs_tstorm"] + 0.2 * X["er_max_gust_kt"] + rng.normal(0, 0.5, n)
    return wx.LGBModel(wx.FLIGHT_FEATURES).fit(X, y.to_numpy(), rounds=120), X


def test_lgbm_drivers_are_sorted_labelled_and_carry_values():
    model, X = storm_model()
    row = X[X["o_obs_tstorm"] == 1].head(1)
    drivers = ex.lgbm_drivers(model, row, k=3)[0]
    assert len(drivers) == 3
    assert drivers[0]["feature"] == "o_obs_tstorm" and drivers[0]["contribution_min"] > 5
    assert abs(drivers[0]["contribution_min"]) >= abs(drivers[1]["contribution_min"]) >= abs(drivers[2]["contribution_min"])
    assert drivers[0]["value"] == 1.0 and "thunderstorm" in drivers[0]["label"].lower() and "origin" in drivers[0]["label"].lower()


def test_every_model_feature_has_a_readable_label():
    from aerogrid.ml import congestion as cg

    for f in set(wx.FLIGHT_FEATURES) | set(cg.AIRPORT_FEATURES):
        label = ex.label(f)
        assert isinstance(label, str) and label and label != f, f
    assert ex.label("zone_monsoon") == "weather zone: monsoon"
    assert "destination" in ex.label("d_fc_gust_kt").lower()


def test_gcn_drivers_find_the_feature_the_prediction_depends_on_by_ablation():
    n, f = 6, 4
    ahat = gcn.operator_inflow(sparse.csr_matrix(np.ones((n, n)) - np.eye(n)))
    rng = np.random.default_rng(1)
    x = rng.normal(size=(300, n, f))
    teacher = gcn.GCN(f, hidden=8, seed=4)
    teacher.W1[:] = 0.0
    teacher.W1[2, :] = 1.0  # only feature 2 matters
    teacher.W2[:] = 2.0
    y = teacher.forward(ahat, x)[0]
    model = gcn.GCN(f, hidden=8, seed=0)
    model.fit(ahat, x, y, np.ones((300, n)), np.ones((300, n)), val=None, scaler=gcn.Standardizer().fit(x), epochs=120, batch=50, lr=1e-2, patience=50)
    names = ["a", "b", "c", "d"]
    drivers = ex.gcn_drivers(model, ahat, x[0], node=1, head=0, feature_names=names, k=2)
    assert drivers[0]["feature"] == "c"
    assert abs(drivers[0]["contribution_min"]) > abs(drivers[1]["contribution_min"])
    assert drivers[0]["method"] == "ablation to the training mean"


def test_gcn_ablation_contributions_are_relative_to_the_unablated_prediction():
    ahat = sparse.identity(3, format="csr")
    model = gcn.GCN(2, hidden=4, seed=0)
    model.scaler = gcn.Standardizer().fit(np.random.default_rng(0).normal(size=(10, 3, 2)))
    x = np.random.default_rng(2).normal(size=(3, 2))
    base = model.predict(ahat, x)[0, 0]
    d = ex.gcn_drivers(model, ahat, x, node=0, head=0, feature_names=["p", "q"], k=2)
    xs = x.copy()
    xs[:, 0] = model.scaler.mean[0]
    assert {r["feature"]: r["contribution_min"] for r in d}["p"] == pytest.approx(base - model.predict(ahat, xs)[0, 0])


def test_cyclic_pairs_and_same_label_features_are_merged_into_one_driver():
    rng = np.random.default_rng(0)
    n = 1200
    X = pd.DataFrame(0.0, index=range(n), columns=wx.FLIGHT_FEATURES)
    for c in ("month_sin", "month_cos", "hour_sin", "hour_cos"):
        X[c] = rng.normal(size=n)
    y = 5 * X["month_sin"] + 3 * X["month_cos"] + 4 * X["hour_cos"] + 30
    model = wx.LGBModel(wx.FLIGHT_FEATURES).fit(X, y.to_numpy(), rounds=150)
    row = X.iloc[[0]]
    merged = ex.lgbm_drivers(model, row, k=3)[0]
    labels = [d["label"] for d in merged]
    assert len(labels) == len(set(labels)), "no label appears twice"
    season = next(d for d in merged if d["label"] == "season (month)")
    assert season["feature"] == "month_sin+month_cos" and season["value"] is None
    raw = model.contrib(row)[0][:-1]
    names = wx.FLIGHT_FEATURES
    assert season["contribution_min"] == pytest.approx(raw[names.index("month_sin")] + raw[names.index("month_cos")])
    unmerged = ex.lgbm_drivers(model, row, k=3, merge=False)[0]
    assert len(unmerged) == 3 and all(d["value"] is not None for d in unmerged)
