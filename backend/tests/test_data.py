import math

import pytest
from scipy.sparse.csgraph import connected_components

from aerogrid import data

ZONES = {"tropical", "subtropical", "temperate", "subpolar", "monsoon"}


def test_load_airports_matches_seed_and_has_no_unresolved_rows():
    airports = data.load_airports()
    assert len(airports) == 155
    assert all(a.notes == "" for a in airports), "rows marked `?` must be resolved against OurAirports"
    jfk = data.get_airport("jfk")
    assert (jfk.icao, jfk.country, jfk.tier) == ("KJFK", "US", 1)
    assert data.get_airport("EGLL").iata == "LHR"


def test_get_airport_unknown_raises_keyerror():
    with pytest.raises(KeyError):
        data.get_airport("ZZZ")


def test_great_circle_km():
    jfk, lhr = data.get_airport("JFK"), data.get_airport("LHR")
    assert data.great_circle_km(jfk.lat, jfk.lon, lhr.lat, lhr.lon) == pytest.approx(5540, rel=0.01)
    assert data.great_circle_km(10, 20, 10, 20) == 0


def test_great_circle_point_interpolates_along_the_path():
    lat, lon = data.great_circle_point(40.0, -74.0, 51.0, 0.0, 0.0)
    assert (round(lat, 6), round(lon, 6)) == (40.0, -74.0)
    lat, lon = data.great_circle_point(40.0, -74.0, 51.0, 0.0, 1.0)
    assert (round(lat, 6), round(lon, 6)) == (51.0, 0.0)
    mid = data.great_circle_point(40.0, -74.0, 51.0, 0.0, 0.5)
    d1 = data.great_circle_km(40.0, -74.0, *mid)
    d2 = data.great_circle_km(*mid, 51.0, 0.0)
    assert d1 == pytest.approx(d2, rel=1e-6)


def test_initial_bearing():
    jfk, lhr = data.get_airport("JFK"), data.get_airport("LHR")
    assert data.initial_bearing(jfk.lat, jfk.lon, lhr.lat, lhr.lon) == pytest.approx(51.2, abs=0.5)
    assert data.initial_bearing(0, 0, 10, 0) == pytest.approx(0)
    assert data.initial_bearing(0, 0, 0, 10) == pytest.approx(90)
    assert data.initial_bearing(0, 0, -10, 0) == pytest.approx(180)
    assert data.initial_bearing(0, 0, 0, -10) == pytest.approx(270)


@pytest.mark.parametrize(
    "iata,month,zone",
    [
        ("SIN", 1, "tropical"),
        ("SIN", 7, "tropical"),
        ("HNL", 4, "tropical"),
        ("DXB", 7, "subtropical"),
        ("CAI", 1, "subtropical"),
        ("SYD", 1, "subtropical"),
        ("JFK", 1, "temperate"),
        ("LHR", 7, "temperate"),
        ("MEL", 1, "temperate"),
        ("KEF", 1, "subpolar"),
        ("ANC", 7, "subpolar"),
        ("OSL", 12, "subpolar"),
        ("DEL", 7, "monsoon"),
        ("DEL", 1, "subtropical"),
        ("BOM", 6, "monsoon"),
        ("BOM", 12, "tropical"),
        ("BKK", 8, "monsoon"),
        ("BKK", 1, "tropical"),
        ("HKG", 6, "monsoon"),
        ("CMB", 7, "monsoon"),
        ("MLE", 7, "monsoon"),
    ],
)
def test_weather_zone_known_airports(iata, month, zone):
    a = data.get_airport(iata)
    assert data.weather_zone(a.lat, a.lon, month) == zone


def test_weather_zone_covers_all_five_zones_over_the_seed():
    seen = {data.weather_zone(a.lat, a.lon, m) for a in data.load_airports() for m in range(1, 13)}
    assert seen == ZONES


@pytest.mark.parametrize("args", [(0, 0, 0), (0, 0, 13), (91, 0, 1), (-91, 0, 1), (0, 181, 1)])
def test_weather_zone_rejects_invalid_input(args):
    with pytest.raises(ValueError):
        data.weather_zone(*args)


def test_routes_graph_is_symmetric_connected_and_labelled_derived():
    g = data.build_routes_graph()
    assert g.origin == "derived"
    assert len(g.codes) == 155
    a = g.adjacency
    assert (a != a.T).nnz == 0
    assert a.diagonal().sum() == 0
    assert connected_components(a, directed=False)[0] == 1


def test_routes_graph_tier1_hubs_form_a_mesh_and_weights_are_distances():
    g = data.build_routes_graph()
    hubs = [a.iata for a in data.load_airports() if a.tier == 1]
    for h in hubs:
        assert set(hubs) - {h} <= set(g.neighbors(h))
    jfk, lhr = data.get_airport("JFK"), data.get_airport("LHR")
    assert g.distance_km("JFK", "LHR") == pytest.approx(data.great_circle_km(jfk.lat, jfk.lon, lhr.lat, lhr.lon))
    assert g.degree("JFK") == len(g.neighbors("JFK"))


def test_routes_graph_every_airport_has_a_route_and_is_deterministic():
    g1, g2 = data.build_routes_graph(), data.build_routes_graph()
    assert all(g1.degree(c) >= 2 for c in g1.codes)
    assert (g1.adjacency != g2.adjacency).nnz == 0
    assert g1.codes == g2.codes


def test_routes_graph_accepts_real_edges_and_labels_them():
    g = data.build_routes_graph(edges=[("JFK", "LHR"), ("LHR", "CDG")], origin="openflights")
    assert g.origin == "openflights"
    assert g.neighbors("LHR") == ["CDG", "JFK"]
    assert g.degree("SIN") == 0


def test_routes_graph_rejects_unknown_airport_in_edges():
    with pytest.raises(ValueError):
        data.build_routes_graph(edges=[("JFK", "ZZZ")], origin="openflights")


def test_default_routes_graph_is_cached():
    assert data.routes_graph() is data.routes_graph()
    assert not math.isnan(data.routes_graph().adjacency.sum())
