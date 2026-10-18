import numpy as np
import pandas as pd
import pytest

from aerogrid.ml import bird


def ts(s):
    return pd.Timestamp(s, tz="UTC").timestamp()


def test_solar_elevation_sanity():
    assert bird.solar_elevation(0.0, 0.0, ts("2026-03-20 12:00")) == pytest.approx(90, abs=3)
    assert bird.solar_elevation(0.0, 0.0, ts("2026-03-20 00:00")) == pytest.approx(-90, abs=3)
    assert bird.solar_elevation(60.0, 0.0, ts("2026-12-21 12:00")) < 10  # low winter sun
    assert bird.solar_elevation(60.0, 0.0, ts("2026-06-21 12:00")) > 50  # high summer sun


def risk(lat, lon, when, **kw):
    return bird.bird_risk(lat, lon, ts(when), **kw)


def test_label_and_ranges():
    r = risk(45, -75, "2026-04-25 03:00")
    assert r["label"] == "prior-based estimate" and r["source"] == "prior"
    assert 0 <= r["risk_index"] <= 1 and r["risk_index"] == r["prior_index"]
    lo, hi = r["altitude_band_ft"]
    assert 0 <= lo < hi
    assert r["flyway"] == "Atlantic Americas"


def test_migration_peaks_in_spring_and_autumn_not_midsummer_or_winter():
    spring = risk(45, -75, "2026-04-25 03:00")["risk_index"]
    autumn = risk(45, -75, "2026-10-01 03:00")["risk_index"]
    summer = risk(45, -75, "2026-07-10 03:00")["risk_index"]
    winter = risk(45, -75, "2026-01-15 03:00")["risk_index"]
    assert spring > 1.5 * summer and autumn > 1.5 * summer and spring > 1.5 * winter


def test_southern_hemisphere_is_mirrored():
    # SH spring migration peaks around Oct-Nov and autumn around Mar-Apr, so April is high and July is low
    assert risk(-35, 150, "2026-04-05 03:00")["risk_index"] > 1.5 * risk(-35, 150, "2026-07-10 03:00")["risk_index"]


def test_night_beats_day_and_the_night_band_is_taller():
    night, day = risk(45, -75, "2026-04-25 05:00"), risk(45, -75, "2026-04-25 17:00")  # 00:00 vs 12:00 local solar
    assert night["risk_index"] > day["risk_index"]
    assert night["altitude_band_ft"][1] > day["altitude_band_ft"][1]


def test_tropics_have_weak_seasonality():
    a, b = risk(2, 100, "2026-04-25 03:00")["risk_index"], risk(2, 100, "2026-07-10 03:00")["risk_index"]
    assert max(a, b) / min(a, b) < 1.5


def test_ebird_abundance_blends_only_when_it_is_real_data():
    base = risk(45, -75, "2026-04-25 03:00")
    busy = risk(45, -75, "2026-04-25 03:00", ebird={"source": "live", "individuals": 5000})
    quiet = risk(45, -75, "2026-04-25 03:00", ebird={"source": "cached", "individuals": 0})
    fake = risk(45, -75, "2026-04-25 03:00", ebird={"source": "simulated", "individuals": 5000})
    assert busy["risk_index"] > base["risk_index"] > quiet["risk_index"]
    assert busy["source"] == "prior+ebird" and busy["prior_index"] == base["prior_index"]
    assert fake["risk_index"] == base["risk_index"] and fake["source"] == "prior"
    assert busy["risk_index"] <= 1.0


@pytest.mark.parametrize("lat,lon,name", [(40, -74, "Atlantic Americas"), (41, -90, "Mississippi Americas"), (47, -122, "Pacific Americas"), (51, 0, "East Atlantic"), (35, 139, "East Asian-Australasian"), (0, 25, "unclassified")])
def test_flyway_labels(lat, lon, name):
    assert bird.flyway(lat, lon) == name


def test_vectorised_prior_matches_scalar():
    lat = np.array([45.0, -35.0, 2.0])
    lon = np.array([-75.0, 150.0, 100.0])
    t = np.array([ts("2026-04-25 03:00"), ts("2026-04-05 03:00"), ts("2026-07-10 03:00")])
    vec = bird.prior_index(lat, lon, t)
    assert np.allclose(vec, [bird.bird_risk(a, b, c)["prior_index"] for a, b, c in zip(lat, lon, t)], atol=1e-4)  # bird_risk rounds to 4 dp


def test_flight_bird_index_averages_both_ends():
    o = (45.0, -75.0, ts("2026-04-25 03:00"))
    d = (45.0, -75.0, ts("2026-07-10 03:00"))
    got = bird.flight_bird_index(*o[:2], o[2], *d[:2], d[2])
    assert got == pytest.approx((bird.prior_index(*map(np.array, o))[()] + bird.prior_index(*map(np.array, d))[()]) / 2)
