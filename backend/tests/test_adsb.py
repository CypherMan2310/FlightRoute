import json

import httpx

from aerogrid import data
from aerogrid.sources import adsb

from conftest import fixture_json, json_response, mock_client

JFK = (40.64, -73.78)
ENABLED = {"OPENSKY_ENABLED": "1", "OPENSKY_CLIENT_ID": "my-client", "OPENSKY_CLIENT_SECRET": "very-secret"}


class Upstream:
    """Routes requests by host and counts them."""

    def __init__(self, adsblol=None, opensky=None, token=None):
        self.adsblol, self.opensky, self.token = adsblol, opensky, token
        self.calls = {"adsb.lol": [], "opensky": [], "token": []}

    def __call__(self, request: httpx.Request) -> httpx.Response:
        host = request.url.host
        if host == "api.adsb.lol":
            self.calls["adsb.lol"].append(request)
            return self.adsblol(request) if self.adsblol else httpx.Response(503)
        if host == "opensky-network.org":
            self.calls["opensky"].append(request)
            return self.opensky(request) if self.opensky else httpx.Response(503)
        if host == "auth.opensky-network.org":
            self.calls["token"].append(request)
            return self.token(request) if self.token else json_response("opensky_token.constructed.json")
        raise AssertionError(f"unexpected host {host}")


def adsblol_ok(request):
    return json_response("adsblol_point.json")


def opensky_ok(request):
    return json_response("opensky_states.json")


def test_adsblol_live_is_normalised(cache):
    up = Upstream(adsblol=adsblol_ok)
    out = adsb.get_flights(*JFK, 100, client=mock_client(up), cache=cache, env={})
    assert (out["source"], out["provider"], out["count"]) == ("live", "adsb.lol", 7)
    assert up.calls["adsb.lol"][0].url.path == "/v2/point/40.64/-73.78/100"
    ground = out["flights"][0]
    assert (ground["icao24"], ground["callsign"], ground["on_ground"], ground["alt_ft"]) == ("a2b17a", "UAL1701", True, 0)
    assert ground["aircraft_type"] == "B38M" and ground["track_deg"] is None
    air = out["flights"][2]
    assert (air["on_ground"], air["alt_ft"], air["gs_kt"], air["vrate_fpm"]) == (False, 1675, 134.5, -384)
    assert set(air) == {"icao24", "callsign", "lat", "lon", "alt_ft", "on_ground", "gs_kt", "track_deg", "vrate_fpm", "aircraft_type"}


def test_adsblol_radius_is_clamped_to_250_nm(cache):
    up = Upstream(adsblol=adsblol_ok)
    adsb.get_flights(*JFK, 900, client=mock_client(up), cache=cache, env={})
    assert up.calls["adsb.lol"][0].url.path.endswith("/250")


def test_second_call_is_served_from_cache_then_refreshed_after_15_s(cache, clock):
    up = Upstream(adsblol=adsblol_ok)
    client = mock_client(up)
    adsb.get_flights(*JFK, 100, client=client, cache=cache, env={})
    clock.advance(10)
    again = adsb.get_flights(*JFK, 100, client=client, cache=cache, env={})
    assert (again["source"], again["age_s"], len(up.calls["adsb.lol"])) == ("cached", 10, 1)
    clock.advance(6)
    assert adsb.get_flights(*JFK, 100, client=client, cache=cache, env={})["source"] == "live"
    assert len(up.calls["adsb.lol"]) == 2


def test_dead_upstream_degrades_to_simulated_not_an_exception(cache):
    out = adsb.get_flights(*JFK, 100, client=mock_client(Upstream()), cache=cache, env={})
    assert out["source"] == "simulated" and out["provider"] == "simulated"
    assert out["fallback_reason"] == "adsb.lol:http_503"
    assert out["count"] == len(out["flights"])


def test_timeout_degrades_to_simulated(cache):
    def slow(request):
        raise httpx.ReadTimeout("slow", request=request)

    out = adsb.get_flights(*JFK, 100, client=mock_client(Upstream(adsblol=slow)), cache=cache, env={})
    assert (out["source"], out["fallback_reason"]) == ("simulated", "adsb.lol:timeout")


def test_garbage_payload_degrades_to_simulated(cache):
    up = Upstream(adsblol=lambda r: httpx.Response(200, text="<html>nope</html>"))
    out = adsb.get_flights(*JFK, 100, client=mock_client(up), cache=cache, env={})
    assert (out["source"], out["fallback_reason"]) == ("simulated", "adsb.lol:bad_payload")


def test_opensky_is_off_by_default_even_when_adsblol_is_down(cache):
    up = Upstream()
    adsb.get_flights(*JFK, 100, client=mock_client(up), cache=cache, env={"OPENSKY_CLIENT_ID": "x", "OPENSKY_CLIENT_SECRET": "y"})
    assert up.calls["opensky"] == [] and up.calls["token"] == []


def test_opensky_fallback_when_enabled_uses_oauth_and_normalises(cache):
    up = Upstream(opensky=opensky_ok)
    out = adsb.get_flights(*JFK, 100, client=mock_client(up), cache=cache, env=ENABLED)
    assert (out["source"], out["provider"]) == ("live", "opensky")
    req = up.calls["opensky"][0]
    assert req.headers["authorization"] == "Bearer test-token"
    assert {"lamin", "lomin", "lamax", "lomax"} <= set(req.url.params)
    token_req = up.calls["token"][0]
    assert b"client_credentials" in token_req.content and b"my-client" in token_req.content
    by_id = {f["icao24"]: f for f in out["flights"]}
    f = by_id["a703e4"]
    assert (f["callsign"], f["on_ground"], f["alt_ft"], f["gs_kt"], f["track_deg"], f["vrate_fpm"]) == ("N551HU", False, 2475, 144.7, 130.52, 0)
    assert by_id["a35254"]["on_ground"] is True and by_id["a35254"]["alt_ft"] == 0
    assert all(data.great_circle_km(*JFK, f["lat"], f["lon"]) <= 100 * 1.852 + 1 for f in out["flights"])
    assert "very-secret" not in json.dumps(out)


def test_opensky_without_credentials_is_anonymous(cache):
    up = Upstream(opensky=opensky_ok)
    adsb.get_flights(*JFK, 100, client=mock_client(up), cache=cache, env={"OPENSKY_ENABLED": "1"})
    assert up.calls["token"] == [] and "authorization" not in up.calls["opensky"][0].headers


def test_opensky_token_is_reused_across_calls(cache):
    up = Upstream(opensky=opensky_ok)
    client = mock_client(up)
    adsb.get_flights(40.64, -73.78, 100, client=client, cache=cache, env=ENABLED)
    adsb.get_flights(40.70, -73.90, 100, client=client, cache=cache, env=ENABLED)
    assert len(up.calls["opensky"]) == 2 and len(up.calls["token"]) == 1


def test_opensky_429_pauses_opensky_for_the_advertised_time(cache, clock):
    up = Upstream(opensky=lambda r: httpx.Response(429, headers={"X-Rate-Limit-Retry-After-Seconds": "120"}))
    client = mock_client(up)
    first = adsb.get_flights(40.64, -73.78, 100, client=client, cache=cache, env=ENABLED)
    assert first["source"] == "simulated" and "opensky:http_429" in first["fallback_reason"]
    adsb.get_flights(40.70, -73.90, 100, client=client, cache=cache, env=ENABLED)
    assert len(up.calls["opensky"]) == 1
    clock.advance(121)
    adsb.get_flights(40.80, -74.00, 100, client=client, cache=cache, env=ENABLED)
    assert len(up.calls["opensky"]) == 2


def test_fixture_matches_documented_opensky_field_order():
    row = fixture_json("opensky_states.json")["states"][0]
    assert len(row) == 17 and isinstance(row[5], float) and isinstance(row[6], float)
