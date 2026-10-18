import httpx
import pytest

from aerogrid.sources import metar

from conftest import fixture_json, json_response, mock_client


def recorder(response):
    calls = []

    def handler(request):
        calls.append(request)
        return response(request) if callable(response) else response

    return handler, calls


def test_metar_live_is_normalised(cache):
    handler, calls = recorder(lambda r: json_response("metar_kjfk_kord_vidp.json"))
    out = metar.get_metar(["KJFK", "KORD", "VIDP"], client=mock_client(handler), cache=cache)
    assert (out["source"], out["provider"], out["missing"]) == ("live", "aviationweather.gov", [])
    req = calls[0]
    assert req.url.host == "aviationweather.gov" and req.url.path == "/api/data/metar"
    assert req.url.params["ids"] == "KJFK,KORD,VIDP" and req.url.params["format"] == "json"
    jfk = out["stations"]["KJFK"]
    assert jfk["source"] == "live"
    assert (jfk["temp_c"], jfk["dewpoint_c"], jfk["wind_dir_deg"], jfk["wind_kt"], jfk["gust_kt"]) == (13.3, 8.3, 330, 15, 23)
    assert (jfk["visibility_sm"], jfk["altimeter_hpa"], jfk["wx"], jfk["flight_category"]) == (10.0, 1012.3, "-RA", "VFR")
    assert jfk["clouds"] == [{"cover": "OVC", "base_ft": 10000}] and jfk["ceiling_ft"] == 10000
    assert jfk["obs_time"] == 1792219860 and jfk["raw"].startswith("METAR KJFK")
    vidp = out["stations"]["VIDP"]
    assert (vidp["visibility_sm"], vidp["flight_category"], vidp["ceiling_ft"], vidp["gust_kt"]) == (2.49, "IFR", None, None)


def test_flight_category_is_computed_when_upstream_omits_it(cache):
    rows = fixture_json("metar_kjfk_kord_vidp.json")
    for r in rows:
        r.pop("fltCat")
    handler, _ = recorder(lambda r: httpx.Response(200, json=rows))
    out = metar.get_metar(["KJFK", "VIDP"], client=mock_client(handler), cache=cache)
    assert out["stations"]["KJFK"]["flight_category"] == "VFR"
    assert out["stations"]["VIDP"]["flight_category"] == "IFR"


def test_variable_wind_has_no_direction(cache):
    rows = fixture_json("metar_kjfk_kord_vidp.json")[:1]
    rows[0]["wdir"] = "VRB"
    handler, _ = recorder(lambda r: httpx.Response(200, json=rows))
    assert metar.get_metar(["KJFK"], client=mock_client(handler), cache=cache)["stations"]["KJFK"]["wind_dir_deg"] is None


def test_stations_without_a_metar_are_reported_missing_not_invented(cache):
    handler, _ = recorder(lambda r: json_response("metar_kjfk_kord_vidp.json"))
    out = metar.get_metar(["KJFK", "KZZZ"], client=mock_client(handler), cache=cache)
    assert out["missing"] == ["KZZZ"] and list(out["stations"]) == ["KJFK"]
    assert out["source"] == "live"


def test_only_uncached_stations_go_upstream_and_ttl_is_ten_minutes(cache, clock):
    handler, calls = recorder(lambda r: json_response("metar_kjfk_kord_vidp.json"))
    client = mock_client(handler)
    metar.get_metar(["KJFK"], client=client, cache=cache)
    clock.advance(300)
    out = metar.get_metar(["KJFK", "KORD"], client=client, cache=cache)
    assert calls[1].url.params["ids"] == "KORD"
    assert (out["stations"]["KJFK"]["source"], out["stations"]["KORD"]["source"], out["source"]) == ("cached", "live", "cached")
    clock.advance(301)
    assert metar.get_metar(["KJFK"], client=client, cache=cache)["stations"]["KJFK"]["source"] == "live"


def test_dead_upstream_simulates_every_station(cache):
    handler, _ = recorder(httpx.Response(500))
    out = metar.get_metar(["KJFK", "EGLL"], client=mock_client(handler), cache=cache)
    assert (out["source"], out["fallback_reason"]) == ("simulated", "http_500")
    assert {s["source"] for s in out["stations"].values()} == {"simulated"}
    assert out["stations"]["EGLL"]["flight_category"] in {"VFR", "MVFR", "IFR", "LIFR"}


def test_unknown_station_cannot_be_simulated_and_is_missing(cache):
    handler, _ = recorder(httpx.Response(500))
    out = metar.get_metar(["KJFK", "XXXX"], client=mock_client(handler), cache=cache)
    assert out["missing"] == ["XXXX"] and list(out["stations"]) == ["KJFK"]


def test_no_content_answer_is_a_failure_not_live_data(cache):
    handler, _ = recorder(httpx.Response(204))
    out = metar.get_metar(["KJFK"], client=mock_client(handler), cache=cache)
    assert (out["source"], out["fallback_reason"]) == ("simulated", "empty_response")


def test_stale_cache_beats_simulation(cache, clock):
    ok, _ = recorder(lambda r: json_response("metar_kjfk_kord_vidp.json"))
    metar.get_metar(["KJFK"], client=mock_client(ok), cache=cache)
    clock.advance(900)
    dead, _ = recorder(httpx.Response(503))
    out = metar.get_metar(["KJFK"], client=mock_client(dead), cache=cache)
    assert out["source"] == "cached" and out["stations"]["KJFK"]["stale"] is True


def test_station_ids_are_normalised_and_limited(cache):
    handler, calls = recorder(lambda r: json_response("metar_kjfk_kord_vidp.json"))
    metar.get_metar(["kjfk", " kord "], client=mock_client(handler), cache=cache)
    assert calls[0].url.params["ids"] == "KJFK,KORD"
    with pytest.raises(ValueError):
        metar.get_metar([f"K{i:03d}" for i in range(101)], client=mock_client(handler), cache=cache)
    with pytest.raises(ValueError):
        metar.get_metar([], client=mock_client(handler), cache=cache)


def test_taf_live_is_normalised(cache):
    handler, calls = recorder(lambda r: json_response("taf_kjfk.json"))
    out = metar.get_taf(["KJFK"], client=mock_client(handler), cache=cache)
    assert calls[0].url.path == "/api/data/taf" and out["source"] == "live"
    taf = out["stations"]["KJFK"]
    assert taf["raw"].startswith("TAF KJFK") and taf["valid_from"] == 1792216800
    first, second = taf["periods"][0], taf["periods"][1]
    assert (first["wind_dir_deg"], first["wind_kt"], first["gust_kt"], first["wx"], first["visibility_sm"]) == (340, 14, 25, "VCSH", 6.0)
    assert first["ceiling_ft"] == 8000 and first["flight_category"] == "VFR"
    assert second["change"] == "FM" and first["from"] < first["to"]


def test_taf_dead_upstream_simulates(cache):
    handler, _ = recorder(httpx.Response(503))
    out = metar.get_taf(["KJFK"], client=mock_client(handler), cache=cache)
    assert (out["source"], out["stations"]["KJFK"]["source"]) == ("simulated", "simulated")
    assert len(out["stations"]["KJFK"]["periods"]) >= 2


def test_a_station_that_never_reports_does_not_degrade_the_batch_or_get_refetched(cache):
    handler, calls = recorder(lambda r: json_response("metar_kjfk_kord_vidp.json"))
    client = mock_client(handler)
    first = metar.get_metar(["KJFK", "KZZZ"], client=client, cache=cache)
    second = metar.get_metar(["KJFK", "KZZZ"], client=client, cache=cache)
    assert first["missing"] == second["missing"] == ["KZZZ"]
    assert (first["source"], second["source"]) == ("live", "cached") and len(calls) == 1
    only_the_ghost = metar.get_metar(["KZZZ"], client=client, cache=cache)
    assert only_the_ghost["missing"] == ["KZZZ"] and len(calls) == 1, "a known-missing station alone is not an upstream request"
