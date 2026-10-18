import numpy as np
import pandas as pd
import pytest

from aerogrid.ml import ensemble as en


def carrier_frame():
    rng = np.random.default_rng(0)
    rows = []
    for carrier, origin, month, mean, n in [("AA", "JFK", 1, 10.0, 60), ("AA", "JFK", 2, 4.0, 60), ("AA", "ORD", 1, 7.0, 5), ("DL", "JFK", 1, 2.0, 80), ("DL", "ATL", 3, 3.0, 80)]:
        ts = pd.Timestamp(f"2025-{month:02d}-15 12:00", tz="UTC").timestamp()
        for v in rng.normal(mean, 0.01, n):
            rows.append({"carrier": carrier, "origin": origin, "ts_dep": ts, "carrier_delay": v, "security_delay": 0.0})
    return pd.DataFrame(rows)


def _jan():
    return pd.Timestamp("2026-01-10", tz="UTC").timestamp()


def _q(carrier, origin):
    return pd.DataFrame({"carrier": [carrier], "origin": [origin], "ts_dep": [_jan()]})


def test_without_shrinkage_the_deepest_available_cell_mean_is_used():
    m = en.CarrierModel(shrink=0.0).fit(carrier_frame())
    assert m.predict(_q("AA", "JFK"))[0] == pytest.approx(10.0, abs=0.05)  # carrier x origin x month
    assert m.predict(_q("DL", "JFK"))[0] == pytest.approx(2.0, abs=0.05)
    assert m.predict(_q("AA", "SEA"))[0] == pytest.approx(np.mean([10.0] * 60 + [4.0] * 60 + [7.0] * 5), abs=0.1)  # unseen origin: carrier-wide mean
    assert m.predict(_q("ZZ", "JFK"))[0] == pytest.approx(m.global_mean)  # unseen carrier: global mean


def test_hierarchical_shrinkage_pulls_small_cells_toward_their_parent():
    k = 100.0
    m = en.CarrierModel(shrink=k).fit(carrier_frame())
    g = m.global_mean
    n_c, mean_c = 125, np.mean([10.0] * 60 + [4.0] * 60 + [7.0] * 5)  # AA overall
    carrier = (n_c * mean_c + k * g) / (n_c + k)
    n_o, mean_o = 120, np.mean([10.0] * 60 + [4.0] * 60)  # AA at JFK
    origin = (n_o * mean_o + k * carrier) / (n_o + k)
    cell = (60 * 10.0 + k * origin) / (60 + k)  # AA at JFK in January
    assert m.predict(_q("AA", "JFK"))[0] == pytest.approx(cell, abs=0.02)
    assert m.predict(_q("AA", "JFK"))[0] < 10.0, "a 60-flight cell is not trusted blindly"
    tiny = (5 * 7.0 + k * carrier) / (5 + k)  # AA at ORD has only 5 flights and, in January, the same 5
    assert m.predict(_q("AA", "ORD"))[0] == pytest.approx(tiny + 0.0, abs=0.5)
    assert abs(m.predict(_q("AA", "ORD"))[0] - carrier) < abs(7.0 - carrier), "5 flights barely move the carrier-wide estimate"


def test_more_data_means_less_shrinkage():
    rng = np.random.default_rng(1)
    rows = []
    for n, origin in ((10, "AAA"), (2000, "BBB")):
        for v in rng.normal(30.0, 1.0, n):
            rows.append({"carrier": "AA", "origin": origin, "ts_dep": _jan(), "carrier_delay": v, "security_delay": 0.0})
    for v in rng.normal(0.0, 1.0, 5000):
        rows.append({"carrier": "AA", "origin": "CCC", "ts_dep": _jan(), "carrier_delay": v, "security_delay": 0.0})
    m = en.CarrierModel(shrink=100.0).fit(pd.DataFrame(rows))
    small, big = m.predict(_q("AA", "AAA"))[0], m.predict(_q("AA", "BBB"))[0]
    assert big > small > m.predict(_q("AA", "CCC"))[0]
    assert big > 25 and small < 20


def test_carrier_explain_reports_the_deepest_cell_and_survives_a_roundtrip(tmp_path):
    m = en.CarrierModel(shrink=50.0).fit(carrier_frame())
    info = m.explain("AA", "JFK", _jan())
    assert info["level"] == "carrier x origin x month" and info["n"] == 60 and info["mean_min"] == pytest.approx(m.predict(_q("AA", "JFK"))[0])
    assert m.explain("AA", "SEA", _jan())["level"] == "carrier" and m.explain("ZZ", "JFK", _jan())["level"] == "global"
    m.save(tmp_path / "carrier.json")
    again = en.CarrierModel.load(tmp_path / "carrier.json")
    assert again.shrink == 50.0 and again.predict(_q("AA", "JFK"))[0] == pytest.approx(m.predict(_q("AA", "JFK"))[0])


def stack_data(n=4000, seed=0, w=(0.8, 0.5, 0.9, 0.4, 0.0, 1.0), base=3.0):
    rng = np.random.default_rng(seed)
    c = np.abs(rng.normal(8, 6, (n, 6)))
    y = base + c @ np.array(w) + rng.normal(0, 0.5, n)
    return pd.DataFrame(c, columns=en.COMPONENTS), y


def test_stacker_recovers_nonnegative_weights_including_a_zero_weight_for_noise():
    c, y = stack_data()
    zones = np.array(["temperate"] * len(y))
    ens = en.Ensemble().fit(c, y, zones, min_rows=500)
    z = ens.zones["temperate"]
    assert z["weights"]["weather"] == pytest.approx(0.8, abs=0.05) and z["weights"]["carrier"] == pytest.approx(1.0, abs=0.05)
    assert z["weights"]["bird"] < 0.05, "a component with no signal must get ~zero weight"
    assert z["baseline"] == pytest.approx(3.0, abs=0.3)
    assert all(v >= 0 for v in z["weights"].values()) and z["baseline"] >= 0


def test_breakdown_is_exactly_additive_and_nonnegative():
    c, y = stack_data(1500)
    ens = en.Ensemble().fit(c, y, np.array(["temperate"] * 1500), min_rows=500)
    total, parts, baseline = ens.predict(c.iloc[:20], np.array(["temperate"] * 20))
    assert np.allclose(total, parts.sum(axis=1) + baseline, atol=1e-9)
    assert (parts >= 0).all() and (baseline >= 0).all()
    one = ens.breakdown(c.iloc[0], "temperate")
    assert one["total_min"] == pytest.approx(total[0]) and set(one["causes"]) == set(en.COMPONENTS)
    assert one["baseline_min"] + sum(one["causes"].values()) == pytest.approx(one["total_min"])


def test_zone_specific_weights_and_pooled_fallback_for_thin_zones():
    ca, ya = stack_data(2000, seed=1, w=(1.0, 0, 0, 0, 0, 0))
    cb, yb = stack_data(2000, seed=2, w=(0, 0, 0, 0, 0, 1.0))
    ct, yt = stack_data(100, seed=3)
    c = pd.concat([ca, cb, ct], ignore_index=True)
    y = np.concatenate([ya, yb, yt])
    zones = np.array(["tropical"] * 2000 + ["temperate"] * 2000 + ["monsoon"] * 100)
    ens = en.Ensemble().fit(c, y, zones, min_rows=1000)
    assert ens.zones["tropical"]["weights"]["weather"] > 0.9 and ens.zones["tropical"]["pooled"] is False
    assert ens.zones["temperate"]["weights"]["carrier"] > 0.9
    assert ens.zones["monsoon"]["pooled"] is True and ens.zones["monsoon"]["n"] == 100
    assert ens.zones["monsoon"]["weights"] == ens.pooled["weights"]
    _, _, _ = ens.predict(c.iloc[:2], np.array(["subpolar", "tropical"]))  # a zone never seen in training uses the pooled weights


def test_ensemble_save_load_roundtrip(tmp_path):
    c, y = stack_data(1200)
    ens = en.Ensemble().fit(c, y, np.array(["temperate"] * 1200), min_rows=500)
    ens.save(tmp_path / "ens.json")
    again = en.Ensemble.load(tmp_path / "ens.json")
    z = np.array(["temperate"] * 5)
    assert np.allclose(again.predict(c.iloc[:5], z)[0], ens.predict(c.iloc[:5], z)[0])


def test_components_are_the_six_causes_in_a_stable_order():
    assert en.COMPONENTS == ("weather", "congestion", "cascade", "route", "bird", "carrier")
