import pytest

from aerogrid import data
from aerogrid.ml import routing

LABEL = "algorithmic (not ML)"


def ap(code):
    a = data.get_airport(code)
    return a.lat, a.lon


def test_module_is_labelled_algorithmic():
    assert routing.LABEL == LABEL


def test_enroute_points_are_at_25_50_75_percent_of_the_great_circle():
    (la, lo), (db, dn) = ap("JFK"), ap("LHR")
    pts = routing.enroute_points(la, lo, db, dn)
    assert len(pts) == 3
    total = data.great_circle_km(la, lo, db, dn)
    fr = [data.great_circle_km(la, lo, *p) / total for p in pts]
    assert fr == pytest.approx([0.25, 0.5, 0.75], abs=1e-6)


def test_direct_flight_within_range_has_no_stop():
    plan = routing.plan_route("JFK", "LAX", "narrowbody")
    assert plan["feasible"] and plan["stops"] == [] and plan["path"] == ["JFK", "LAX"]
    assert plan["detour_pct"] == 0 and plan["label"] == LABEL
    assert plan["direct_km"] == pytest.approx(data.great_circle_km(*ap("JFK"), *ap("LAX")))


def test_narrowbody_transatlantic_needs_a_range_aware_stop_with_small_detour():
    plan = routing.plan_route("JFK", "LHR", "narrowbody")
    usable_km = routing.AIRCRAFT["narrowbody"]["range_nm"] * routing.RANGE_RESERVE * 1.852
    assert plan["feasible"] and len(plan["stops"]) >= 1
    assert all(leg["km"] <= usable_km + 1e-6 for leg in plan["legs"])
    assert 0 < plan["detour_pct"] < 40
    assert plan["total_km"] == pytest.approx(sum(leg["km"] for leg in plan["legs"]))
    assert plan["path"][0] == "JFK" and plan["path"][-1] == "LHR"


def test_widebody_flies_the_same_pair_direct():
    assert routing.plan_route("JFK", "LHR", "widebody")["stops"] == []


def test_stops_never_exceed_the_aircraft_range_and_total_is_at_least_direct():
    plan = routing.plan_route("JFK", "SYD", "widebody")
    usable_km = routing.AIRCRAFT["widebody"]["range_nm"] * routing.RANGE_RESERVE * 1.852
    assert plan["feasible"] and plan["stops"]
    assert all(leg["km"] <= usable_km + 1e-6 for leg in plan["legs"])
    assert plan["total_km"] >= plan["direct_km"]


def test_infeasible_when_no_chain_of_legs_exists():
    plan = routing.plan_route("JFK", "SYD", "regional")
    assert plan["feasible"] is False and plan["path"] == [] and plan["stops"] == []


def test_same_origin_and_destination_or_unknown_inputs_are_rejected():
    with pytest.raises(ValueError):
        routing.plan_route("JFK", "JFK", "narrowbody")
    with pytest.raises(KeyError):
        routing.plan_route("JFK", "ZZZ", "narrowbody")
    with pytest.raises(ValueError):
        routing.plan_route("JFK", "LAX", "zeppelin")


WINDS_W100 = [(100.0, 270.0)] * 3  # 100 kt jet from the west at all three en-route points


def test_tailwind_saves_time_headwind_costs_time_and_only_headwind_counts_as_extra():
    east = routing.wind_extra_minutes(*ap("JFK"), *ap("LHR"), WINDS_W100)
    west = routing.wind_extra_minutes(*ap("LHR"), *ap("JFK"), WINDS_W100)
    assert east["delta_min"] < -30 and east["extra_min"] == 0
    assert 60 < west["extra_min"] < 160 and west["delta_min"] == pytest.approx(west["extra_min"])
    assert west["wind_min"] == pytest.approx(west["nowind_min"] + west["delta_min"])


def test_no_wind_means_no_change():
    r = routing.wind_extra_minutes(*ap("JFK"), *ap("LHR"), [(0.0, 0.0)] * 3)
    assert r["delta_min"] == pytest.approx(0) and r["extra_min"] == 0
    assert r["nowind_min"] == pytest.approx(data.great_circle_km(*ap("JFK"), *ap("LHR")) / 1.852 / routing.DEFAULT_TAS_KT * 60)


def test_crosswind_has_no_effect():
    # due east along the equator (constant 90 deg track), wind from the north: a pure crosswind
    r = routing.wind_extra_minutes(0.0, 0.0, 0.0, 60.0, [(80.0, 0.0)] * 3)
    assert r["delta_min"] == pytest.approx(0, abs=0.5)


def test_vectorised_wind_extra_matches_the_scalar_function():
    import numpy as np

    rng = np.random.default_rng(0)
    pairs = [("JFK", "LHR"), ("LHR", "JFK"), ("LAX", "JFK"), ("DEL", "SIN"), ("SYD", "LAX")]
    o = np.array([ap(a) for a, _ in pairs])
    d = np.array([ap(b) for _, b in pairs])
    speed, wind_from = rng.uniform(0, 120, (3, 5)), rng.uniform(0, 360, (3, 5))
    vec = routing.wind_extra_minutes_vec(o[:, 0], o[:, 1], d[:, 0], d[:, 1], speed, wind_from)
    for i in range(5):
        one = routing.wind_extra_minutes(*o[i], *d[i], [(speed[k, i], wind_from[k, i]) for k in range(3)])
        assert vec["delta_min"][i] == pytest.approx(one["delta_min"]) and vec["extra_min"][i] == pytest.approx(one["extra_min"])
        assert vec["nowind_min"][i] == pytest.approx(one["nowind_min"])
