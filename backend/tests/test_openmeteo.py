from datetime import datetime, timezone

import httpx
import pytest

from aerogrid.sources import openmeteo

from conftest import fixture_json, mock_client

JFK = (40.6398, -73.7789)
LHR = (51.4706, -0.4619)


def serve(payload):
    calls = []

    def handler(request):
        calls.append(request)
        return httpx.Response(200, json=payload)

    return handler, calls


def test_two_points_one_request_normalised(cache):
    handler, calls = serve(fixture_json("openmeteo_two_points.json"))
    out = openmeteo.get_forecast([JFK, LHR], hours=3, client=mock_client(handler), cache=cache)
    assert (out["source"], out["provider"], out["hours"]) == ("live", "open-meteo.com", 3)
    assert len(calls) == 1
    p = calls[0].url.params
    assert (p["latitude"], p["longitude"]) == ("40.64,51.47", "-73.78,-0.46")
    assert (p["forecast_hours"], p["timezone"], p["wind_speed_unit"]) == ("3", "UTC", "kn")
    assert calls[0].url.host == "api.open-meteo.com" and calls[0].url.path == "/v1/forecast"
    jfk, lhr = out["points"]
    assert (jfk["lat"], jfk["lon"]) == JFK and (lhr["lat"], lhr["lon"]) == LHR
    assert len(jfk["hourly"]) == 3 and jfk["source"] == "live"
    h = jfk["hourly"][0]
    assert h["time"] == int(datetime(2026, 10, 17, 8, tzinfo=timezone.utc).timestamp())
    assert (h["temp_c"], h["precip_mm"], h["wind_kt"], h["gust_kt"]) == (11.9, 0.0, 9.5, 24.1)
    assert (h["visibility_m"], h["weather_code"]) == (25000.0, 3)
    hourly = calls[0].url.params["hourly"].split(",")
    assert len(hourly) <= 10, "more than 10 variables would count as extra API calls against the free limit"
    assert (h["cloud_low_pct"], h["cape_jkg"], h["wind250_kt"], h["wind250_dir_deg"]) == (16, 0.0, 92.2, 233)


def test_single_point_answer_is_an_object_not_a_list(cache):
    handler, _ = serve(fixture_json("openmeteo_two_points.json")[0])
    out = openmeteo.get_forecast([JFK], hours=3, client=mock_client(handler), cache=cache)
    assert len(out["points"]) == 1 and out["points"][0]["hourly"][0]["temp_c"] == 11.9


def test_points_are_cached_individually_for_thirty_minutes(cache, clock):
    both = fixture_json("openmeteo_two_points.json")
    handler, calls = serve(both[0])
    client = mock_client(handler)
    openmeteo.get_forecast([JFK], hours=3, client=client, cache=cache)
    handler2, calls2 = serve(both[1])
    out = openmeteo.get_forecast([JFK, LHR], hours=3, client=mock_client(handler2), cache=cache)
    assert calls2[0].url.params["latitude"] == "51.47"
    assert [p["source"] for p in out["points"]] == ["cached", "live"] and out["source"] == "cached"
    clock.advance(1801)
    handler3, calls3 = serve(both)
    openmeteo.get_forecast([JFK, LHR], hours=3, client=mock_client(handler3), cache=cache)
    assert calls3[0].url.params["latitude"] == "40.64,51.47"


def test_dead_upstream_simulates_all_points(cache):
    out = openmeteo.get_forecast([JFK, LHR], hours=4, client=mock_client(lambda r: httpx.Response(502)), cache=cache)
    assert (out["source"], out["fallback_reason"]) == ("simulated", "http_502")
    assert [len(p["hourly"]) for p in out["points"]] == [4, 4]
    assert {p["source"] for p in out["points"]} == {"simulated"}


def test_cached_plus_simulated_is_reported_as_simulated(cache):
    openmeteo.get_forecast([JFK], hours=3, client=mock_client(serve(fixture_json("openmeteo_two_points.json")[0])[0]), cache=cache)
    out = openmeteo.get_forecast([JFK, LHR], hours=3, client=mock_client(lambda r: httpx.Response(500)), cache=cache)
    assert [p["source"] for p in out["points"]] == ["cached", "simulated"]
    assert out["source"] == "simulated"


def test_malformed_answer_simulates(cache):
    handler, _ = serve({"error": True, "reason": "nope"})
    out = openmeteo.get_forecast([JFK], hours=3, client=mock_client(handler), cache=cache)
    assert (out["source"], out["fallback_reason"]) == ("simulated", "bad_payload")


@pytest.mark.parametrize("hours", [0, 169])
def test_hours_out_of_range_is_rejected(cache, hours):
    with pytest.raises(ValueError):
        openmeteo.get_forecast([JFK], hours=hours, cache=cache)


def test_no_points_is_rejected(cache):
    with pytest.raises(ValueError):
        openmeteo.get_forecast([], hours=3, cache=cache)


def test_daily_budget_stops_live_calls_and_degrades_instead_of_exceeding_the_free_limit(cache, monkeypatch):
    monkeypatch.setattr(openmeteo, "DAILY_BUDGET", 3)
    two = fixture_json("openmeteo_two_points.json")
    handler, calls = serve(two)
    client = mock_client(handler)
    first = openmeteo.get_forecast([JFK, LHR], hours=3, client=client, cache=cache)
    assert first["source"] == "live" and len(calls) == 1
    other = openmeteo.get_forecast([(10.0, 10.0), (20.0, 20.0)], hours=3, client=client, cache=cache)
    assert (other["source"], other["fallback_reason"]) == ("simulated", "daily_budget") and len(calls) == 1
    again = openmeteo.get_forecast([JFK, LHR], hours=3, client=client, cache=cache)
    assert again["source"] == "cached" and len(calls) == 1, "cached points never cost budget"
