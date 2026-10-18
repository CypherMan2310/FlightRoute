from datetime import datetime, timezone

import httpx
import pytest

from aerogrid.sources import metno, openmeteo
from aerogrid.sources.cache import USER_AGENT

from conftest import fixture_json, mock_client

ORD = (41.98, -87.90)
JFK = (40.6398, -73.7789)
LHR = (51.4706, -0.4619)
FAST = dict(min_interval_s=0.0)
ORD_KEY = "41.98,-87.90"


def serve(status=200):
    calls = []

    def handler(request):
        calls.append(request)
        return httpx.Response(status, json=fixture_json("metno_complete_ord.json")) if status == 200 else httpx.Response(status)

    return handler, calls


def test_answer_is_normalised_to_the_open_meteo_row_shape_and_says_what_it_lacks(cache):
    handler, calls = serve()
    recs, reason = metno.get_points([ORD], hours=4, client=mock_client(handler), cache=cache, **FAST)
    r = recs[ORD_KEY]
    assert reason is None and (r["source"], r["provider"]) == ("live", "api.met.no") and r["degraded"] == list(metno.DEGRADED)
    assert len(r["hourly"]) == 4, "capped at the requested hours"
    first = r["hourly"][0]
    assert first["time"] == int(datetime(2026, 10, 18, 8, tzinfo=timezone.utc).timestamp())
    assert first["wind_kt"] == pytest.approx(3.8 * 1.943844) and first["temp_c"] == 6.4
    assert (first["precip_mm"], first["cloud_low_pct"]) == (0.0, 0.0)
    assert (first["gust_kt"], first["visibility_m"], first["cape_jkg"], first["wind250_kt"]) == (first["wind_kt"], 10000.0, 0.0, 0.0), "stand-ins, never invented values"
    assert set(first) == set(openmeteo._normalise(fixture_json("openmeteo_two_points.json")[0])["hourly"][0]), "same keys as an Open-Meteo hour"


def test_only_hourly_steps_are_used_the_six_hourly_tail_is_skipped(cache):
    recs, _ = metno.get_points([ORD], hours=48, client=mock_client(serve()[0]), cache=cache, **FAST)
    assert len(recs[ORD_KEY]["hourly"]) == 5


def test_request_identifies_the_application_with_contact_information_and_few_decimals(cache):
    handler, calls = serve()
    metno.get_points([ORD], hours=3, client=mock_client(handler), cache=cache, **FAST)
    req = calls[0]
    assert req.url.host == "api.met.no" and req.url.path == "/weatherapi/locationforecast/2.0/complete"
    assert (req.url.params["lat"], req.url.params["lon"]) == ("41.98", "-87.90")
    assert req.headers["user-agent"].startswith(USER_AGENT) and "github.com/CypherMan2310/FlightRoute" in req.headers["user-agent"]
    handler2, calls2 = serve()
    metno.get_points([LHR], hours=3, client=mock_client(handler2), cache=cache, env={"AEROGRID_CONTACT": "https://example.org/aerogrid"}, **FAST)
    assert calls2[0].headers["user-agent"].endswith("https://example.org/aerogrid")


def test_points_are_cached_for_thirty_minutes(cache, clock):
    handler, calls = serve()
    client = mock_client(handler)
    metno.get_points([ORD], hours=3, client=client, cache=cache, **FAST)
    again, _ = metno.get_points([ORD], hours=3, client=client, cache=cache, **FAST)
    assert len(calls) == 1 and again[ORD_KEY]["source"] == "cached"
    clock.advance(1801)
    metno.get_points([ORD], hours=3, client=client, cache=cache, **FAST)
    assert len(calls) == 2


def test_a_failing_point_is_left_out_never_simulated_here(cache):
    def handler(request):
        return httpx.Response(429) if request.url.params["lat"] == "51.47" else httpx.Response(200, json=fixture_json("metno_complete_ord.json"))

    recs, reason = metno.get_points([ORD, LHR], hours=3, client=mock_client(handler), cache=cache, **FAST)
    assert list(recs) == [ORD_KEY] and reason == "http_429"
    assert cache.get("metno:backoff") is None, "one good answer means the service is up: no back-off"


def test_a_stale_answer_is_reused_and_labelled_when_the_service_fails(cache, clock):
    metno.get_points([ORD], hours=3, client=mock_client(serve()[0]), cache=cache, **FAST)
    clock.advance(1801)
    recs, reason = metno.get_points([ORD], hours=3, client=mock_client(serve(503)[0]), cache=cache, **FAST)
    r = recs[ORD_KEY]
    assert (r["source"], r.get("stale"), r["fallback_reason"], reason) == ("cached", True, "http_503", "http_503")


def test_a_dead_service_stops_the_batch_early_and_is_not_asked_again_for_two_minutes(cache, clock):
    handler, calls = serve(503)
    client = mock_client(handler)
    points = [(30.0 + i, -90.0) for i in range(20)]
    recs, reason = metno.get_points(points, hours=3, client=client, cache=cache, workers=1, **FAST)
    assert recs == {} and reason == "http_503" and len(calls) == metno.GIVE_UP_AFTER, "20 doomed requests become 5"
    again, reason2 = metno.get_points(points, hours=3, client=client, cache=cache, workers=1, **FAST)
    assert again == {} and reason2 == "http_503" and len(calls) == metno.GIVE_UP_AFTER, "back-off: no new request"
    clock.advance(metno.BACKOFF_S + 1)
    metno.get_points(points, hours=3, client=client, cache=cache, workers=1, **FAST)
    assert len(calls) == 2 * metno.GIVE_UP_AFTER


def primary_dead(cache, points, status=429):
    return openmeteo.get_forecast(points, hours=3, client=mock_client(lambda r: httpx.Response(status)), cache=cache)


def test_simulated_points_are_rescued_and_the_payload_keeps_the_reason_open_meteo_was_not_used(cache):
    payload = primary_dead(cache, [JFK, LHR])
    assert (payload["source"], payload["fallback_reason"]) == ("simulated", "http_429")
    out = metno.fill_simulated(payload, [JFK, LHR], 3, client=mock_client(serve()[0]), cache=cache, **FAST)
    assert (out["source"], out["provider"], out["fallback_reason"]) == ("live", "api.met.no", "http_429")
    assert out["degraded"] == sorted(metno.DEGRADED) and "MET Norway" in out["attribution"]
    assert [(p["lat"], p["lon"], p["source"], p["provider"]) for p in out["points"]] == [(*JFK, "live", "api.met.no"), (*LHR, "live", "api.met.no")]
    assert all(len(p["hourly"]) == 3 for p in out["points"])


def test_open_meteo_cached_points_are_kept_and_the_least_real_source_wins(cache):
    openmeteo.get_forecast([JFK], hours=3, client=mock_client(lambda r: httpx.Response(200, json=fixture_json("openmeteo_two_points.json")[0])), cache=cache)
    payload = openmeteo.get_forecast([JFK, LHR], hours=3, client=mock_client(lambda r: httpx.Response(500)), cache=cache)
    assert [p["source"] for p in payload["points"]] == ["cached", "simulated"]
    out = metno.fill_simulated(payload, [JFK, LHR], 3, client=mock_client(serve()[0]), cache=cache, **FAST)
    assert [p["source"] for p in out["points"]] == ["cached", "live"] and out["source"] == "cached"
    assert out["provider"] == "open-meteo.com + api.met.no" and out["points"][0].get("degraded") is None and out["degraded"]


def test_a_healthy_primary_never_calls_met_norway(cache):
    payload = openmeteo.get_forecast([JFK, LHR], hours=3, client=mock_client(lambda r: httpx.Response(200, json=fixture_json("openmeteo_two_points.json"))), cache=cache)
    handler, calls = serve()
    out = metno.fill_simulated(payload, [JFK, LHR], 3, client=mock_client(handler), cache=cache, **FAST)
    assert out is payload and calls == [] and "degraded" not in out


def test_when_both_upstreams_are_down_the_simulation_and_its_reason_stand(cache):
    payload = primary_dead(cache, [JFK], status=503)
    out = metno.fill_simulated(payload, [JFK], 3, client=mock_client(serve(502)[0]), cache=cache, **FAST)
    assert out is payload and (out["source"], out["fallback_reason"]) == ("simulated", "http_503")
