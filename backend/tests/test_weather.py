import numpy as np
import pandas as pd
import pytest

from aerogrid import data
from aerogrid.ml import weather as wx
from aerogrid.sources.metar import _normalise_metar

from conftest import fixture_json


def metar(i):
    return _normalise_metar(fixture_json("metar_kjfk_kord_vidp.json")[i])


def test_obs_features_from_a_real_metar():
    f = wx.obs_features(metar(0))  # KJFK: -RA, OVC100, 10 SM, 15G23KT, VFR
    assert f == {"obs_vis_sm": 10.0, "obs_ceiling_kft": 10.0, "obs_wind_kt": 15.0, "obs_gust_kt": 23.0, "obs_precip": 1.0, "obs_tstorm": 0.0, "obs_cat": 0.0}
    g = wx.obs_features(metar(2))  # VIDP: haze, 2.49 SM, IFR, no gust, no ceiling
    assert (g["obs_vis_sm"], g["obs_ceiling_kft"], g["obs_gust_kt"], g["obs_cat"]) == (2.49, 12.0, 4.0, 2.0)


@pytest.mark.parametrize("code,precip,ts", [("TSRA", 1.0, 1.0), ("+TSRA", 2.0, 1.0), ("+RA", 2.0, 0.0), ("FG", 0.0, 0.0), ("-SN", 1.0, 0.0), ("", 0.0, 0.0)])
def test_obs_precip_and_thunderstorm_flags(code, precip, ts):
    f = wx.obs_features({"wx": code, "visibility_sm": 3, "ceiling_ft": 800, "wind_kt": 10, "gust_kt": None, "flight_category": "IFR"})
    assert (f["obs_precip"], f["obs_tstorm"]) == (precip, ts)


def test_missing_metar_gives_neutral_good_weather():
    assert wx.obs_features(None) == wx.NEUTRAL_OBS
    assert wx.NEUTRAL_OBS["obs_cat"] == 0.0 and wx.NEUTRAL_OBS["obs_tstorm"] == 0.0


def test_fc_features_from_an_hourly_forecast_row():
    f = wx.fc_features({"precip_mm": 1.5, "gust_kt": 30.0, "cloud_low_pct": 80, "visibility_m": 4000.0, "cape_jkg": 1200.0})
    assert f == {"fc_precip_mm": 1.5, "fc_gust_kt": 30.0, "fc_cloud_low_pct": 80.0, "fc_vis_km": 4.0, "fc_cape_jkg": 1200.0}
    assert wx.fc_features({"precip_mm": 0, "gust_kt": 0, "cloud_low_pct": 0, "visibility_m": 50000.0, "cape_jkg": 0})["fc_vis_km"] == 10.0
    assert wx.fc_features(None) == wx.NEUTRAL_FC


def test_calendar_features_use_local_solar_time_and_flag_holidays():
    t = pd.Timestamp("2026-07-04 18:00", tz="UTC").timestamp()  # Saturday, US holiday
    cal = wx.calendar_features([t], [-90.0], 6)
    local_hour = 18 - 6  # lon -90 -> UTC-6 solar time
    assert cal["hour_sin"][0] == pytest.approx(np.sin(2 * np.pi * local_hour / 24))
    assert cal["holiday"][0] == 1.0 and cal["horizon_h"][0] == 6
    assert cal["dow_sin"][0] == pytest.approx(np.sin(2 * np.pi * 5 / 7))
    ordinary = wx.calendar_features([pd.Timestamp("2026-07-08 18:00", tz="UTC").timestamp()], [-90.0], 2)
    assert ordinary["holiday"][0] == 0.0


def _raw_flight_frame(n=3):
    row = {"ts_dep": pd.Timestamp("2026-10-17 15:00", tz="UTC").timestamp(), "horizon_h": 2, "o_lon": -73.8, "o_tier": 1, "d_tier": 2, "o_zone": "temperate", "distance_km": 1000.0, "crs_elapsed_min": 150.0,
           "er_max_gust_kt": 20.0, "er_max_precip_mm": 0.0, "er_max_cape_jkg": 0.0, "er_max_cloud_low_pct": 10.0}
    for side in ("o", "d"):
        row.update({f"{side}_{k}": v for k, v in {**wx.NEUTRAL_OBS, **wx.NEUTRAL_FC}.items()})
    return pd.DataFrame([row] * n)


def test_build_flight_features_has_exactly_the_model_columns():
    x = wx.build_flight_features(_raw_flight_frame())
    assert list(x.columns) == wx.FLIGHT_FEATURES
    assert x.shape[0] == 3 and not x.isna().any().any()
    zone_cols = [c for c in x.columns if c.startswith("zone_")]
    assert x.loc[0, zone_cols].sum() == 1 and x.loc[0, "zone_temperate"] == 1


def test_build_flight_features_rejects_a_missing_raw_column():
    with pytest.raises(KeyError):
        wx.build_flight_features(_raw_flight_frame().drop(columns=["o_obs_tstorm"]))


def _fit_frame(n=1500, seed=0):
    rng = np.random.default_rng(seed)
    raw = pd.concat([_raw_flight_frame(1)] * n, ignore_index=True)
    raw["o_obs_tstorm"] = (rng.random(n) < 0.2).astype(float)
    raw["er_max_gust_kt"] = rng.uniform(5, 50, n)
    raw["ts_dep"] = raw["ts_dep"] + rng.integers(0, 90 * 24, n) * 3600.0
    y = 25 * raw["o_obs_tstorm"] + 0.4 * np.maximum(raw["er_max_gust_kt"] - 30, 0) + rng.normal(0, 1.0, n)
    return wx.build_flight_features(raw), np.maximum(y, 0).to_numpy()


def test_lgb_model_learns_explains_and_roundtrips(tmp_path):
    x, y = _fit_frame()
    model = wx.LGBModel(wx.FLIGHT_FEATURES)
    model.fit(x.iloc[:1200], y[:1200], rounds=150)
    pred = model.predict(x.iloc[1200:])
    assert (pred >= 0).all() and np.corrcoef(pred, y[1200:])[0, 1] > 0.9
    contrib = model.contrib(x.iloc[1200:1210])
    assert contrib.shape == (10, len(wx.FLIGHT_FEATURES) + 1)
    assert np.allclose(contrib.sum(axis=1), model.raw_predict(x.iloc[1200:1210]), atol=1e-6)
    storm = x.iloc[1200:].index[(x.iloc[1200:]["o_obs_tstorm"] == 1).to_numpy()][0]
    assert model.top_drivers(x.loc[[storm]], k=1)[0][0][0] == "o_obs_tstorm"
    model.save(tmp_path / "w")
    again = wx.LGBModel.load(tmp_path / "w")
    assert np.allclose(again.predict(x.iloc[1200:]), pred)


def test_lgb_model_rejects_wrong_columns():
    x, y = _fit_frame(300)
    model = wx.LGBModel(wx.FLIGHT_FEATURES)
    model.fit(x, y, rounds=10)
    with pytest.raises(KeyError):
        model.predict(x.drop(columns=["distance_km"]))


def test_zone_columns_follow_the_data_module():
    assert wx.ZONE_COLS == [f"zone_{z}" for z in data.ZONES]


def test_lgb_early_stopping_stops_well_before_the_round_limit_on_noise():
    rng = np.random.default_rng(0)
    x, _ = _fit_frame(1200)
    y = np.abs(rng.normal(0, 5, len(x)))  # unrelated to the features: validation loss stops improving quickly
    model = wx.LGBModel(wx.FLIGHT_FEATURES).fit(x.iloc[:900], y[:900], val=(x.iloc[900:], y[900:]), rounds=500)
    assert model.booster.num_trees() < 150
