import statistics

import pytest

from aerogrid import data
from aerogrid.sources import simulate as sim
from aerogrid.sources.cache import NoSimulation
from aerogrid.sources.metar import flight_category

NOW = 1_792_221_414.0  # 2026-10-17
JAN = 1_768_000_000.0  # 2026-01-09
JUL = 1_783_600_000.0  # 2026-07-09
APR = 1_776_000_000.0  # 2026-04-12
HOUR = 3600.0
CATEGORIES = {"VFR", "MVFR", "IFR", "LIFR"}


@pytest.mark.parametrize(
    "call",
    [
        lambda: sim.sim_flights(40.64, -73.78, 150, NOW),
        lambda: sim.sim_metar("KJFK", NOW),
        lambda: sim.sim_taf("KJFK", NOW),
        lambda: sim.sim_forecast(40.64, -73.78, NOW, 6),
        lambda: sim.sim_nas(NOW),
        lambda: sim.sim_ebird(40.64, -73.78, NOW),
    ],
)
def test_every_generator_is_deterministic_and_labelled_simulated(call):
    a, b = call(), call()
    assert a == b
    assert a["source"] == "simulated"


def test_sim_metar_is_plausible_and_normalised():
    m = sim.sim_metar("KJFK", NOW)
    assert m["icao"] == "KJFK" and m["raw"] is None
    assert m["flight_category"] in CATEGORIES
    assert 0 <= m["wind_dir_deg"] <= 360 and m["wind_kt"] >= 0
    assert m["visibility_sm"] > 0 and m["dewpoint_c"] <= m["temp_c"]
    assert all(set(c) == {"cover", "base_ft"} for c in m["clouds"])


def test_sim_metar_follows_climate():
    def mean_temp(icao, t0):
        return statistics.mean(sim.sim_metar(icao, t0 + h * HOUR)["temp_c"] for h in range(24))

    assert mean_temp("KJFK", JAN) < mean_temp("OMDB", JAN)
    assert mean_temp("YSSY", JAN) > mean_temp("YSSY", JUL)
    assert mean_temp("KJFK", JUL) > mean_temp("KJFK", JAN)


def test_sim_metar_unknown_station_has_no_simulation():
    with pytest.raises(NoSimulation):
        sim.sim_metar("XXXX", NOW)


def test_sim_metar_changes_over_time():
    assert sim.sim_metar("KJFK", NOW) != sim.sim_metar("KJFK", NOW + 6 * HOUR)


@pytest.mark.parametrize(
    "vis,ceil,cat",
    [(10, None, "VFR"), (10, 5000, "VFR"), (5, None, "MVFR"), (10, 3000, "MVFR"), (2, None, "IFR"), (10, 800, "IFR"), (0.5, None, "LIFR"), (10, 300, "LIFR")],
)
def test_flight_category_thresholds(vis, ceil, cat):
    assert flight_category(vis, ceil) == cat


def test_sim_taf_has_ordered_periods():
    t = sim.sim_taf("KJFK", NOW)
    assert t["icao"] == "KJFK" and len(t["periods"]) >= 2
    assert all(p["from"] < p["to"] for p in t["periods"])
    assert t["periods"][0]["from"] <= t["periods"][1]["from"]
    assert all(p["flight_category"] in CATEGORIES for p in t["periods"])


def test_sim_forecast_shape():
    f = sim.sim_forecast(40.64, -73.78, NOW, 6)
    assert (f["lat"], f["lon"]) == (40.64, -73.78)
    assert len(f["hourly"]) == 6
    assert [h["time"] for h in f["hourly"]] == sorted(h["time"] for h in f["hourly"])
    assert all(h["precip_mm"] >= 0 and h["visibility_m"] > 0 for h in f["hourly"])
    assert all(0 <= h["cloud_low_pct"] <= 100 and h["cape_jkg"] >= 0 for h in f["hourly"])
    assert all(h["wind250_kt"] >= 0 and 0 <= h["wind250_dir_deg"] <= 360 for h in f["hourly"])


def test_sim_flights_stay_inside_the_radius_and_move():
    a = sim.sim_flights(40.64, -73.78, 150, NOW)
    assert a["count"] == len(a["flights"]) > 0
    for f in a["flights"]:
        assert data.great_circle_km(40.64, -73.78, f["lat"], f["lon"]) <= 150 * 1.852 + 1
        assert f["callsign"].startswith("SIM")
    assert len({f["icao24"] for f in a["flights"]}) == a["count"]
    b = sim.sim_flights(40.64, -73.78, 150, NOW + 60)
    moved = {f["icao24"]: (f["lat"], f["lon"]) for f in a["flights"]}
    assert any(moved[f["icao24"]] != (f["lat"], f["lon"]) for f in b["flights"] if f["icao24"] in moved)


def test_sim_nas_lists_only_us_airports_with_valid_statuses():
    seen = set()
    for k in range(30):
        s = sim.sim_nas(NOW + k * 1800)
        for code, st in s["airports"].items():
            assert data.get_airport(code).country in {"US", "PR", "GU"}
            assert st["status"] in {"delay", "ground_delay", "ground_stop"}
            assert st["delay_minutes"] >= 0
            seen.add(st["status"])
    assert seen, "30 half-hour buckets should produce at least one simulated programme"


def test_sim_ebird_has_migration_peaks():
    spring = sim.sim_ebird(45.0, -75.0, APR)
    summer = sim.sim_ebird(45.0, -75.0, JUL)
    assert spring["individuals"] > summer["individuals"] >= 0
    assert spring["n_species"] <= spring["n_records"]
    assert spring["top_species"] == sorted(spring["top_species"], key=lambda s: -s["individuals"])
