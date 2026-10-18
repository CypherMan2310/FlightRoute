import json

import httpx

from aerogrid.sources import ebird

from conftest import fixture_text, mock_client

KEY = "secret-ebird-key"


def serve(status=200, text=None):
    calls = []

    def handler(request):
        calls.append(request)
        return httpx.Response(status, text=fixture_text("ebird_recent.constructed.json") if text is None else text)

    return handler, calls


def test_no_key_means_simulated_without_touching_the_network(cache):
    handler, calls = serve()
    out = ebird.get_recent_observations(40.64, -73.78, client=mock_client(handler), cache=cache, env={})
    assert (out["source"], out["fallback_reason"], calls) == ("simulated", "no_api_key", [])


def test_live_is_summarised_and_key_never_leaks(cache):
    handler, calls = serve()
    out = ebird.get_recent_observations(40.64, -73.78, 25, 14, client=mock_client(handler), cache=cache, env={"EBIRD_API_KEY": KEY})
    req = calls[0]
    assert req.url.host == "api.ebird.org" and req.url.path == "/v2/data/obs/geo/recent"
    assert req.headers["x-ebirdapitoken"] == KEY
    assert (req.url.params["lat"], req.url.params["lng"], req.url.params["dist"], req.url.params["back"]) == ("40.64", "-73.78", "25", "14")
    assert (out["source"], out["provider"], out["n_records"], out["n_species"]) == ("live", "ebird", 6, 4)
    assert out["individuals"] == 120 + 80 + 35 + 2 + 1 + 15  # a record without howMany (presence only) counts as 1
    assert out["top_species"][0] == {"code": "cangoo", "name": "Canada Goose", "individuals": 200}
    assert [s["code"] for s in out["top_species"]] == ["cangoo", "hergul", "osprey", "amerob"]
    assert KEY not in json.dumps(out)


def test_radius_and_window_are_clamped_to_ebird_limits(cache):
    handler, calls = serve()
    ebird.get_recent_observations(40.64, -73.78, 80, 90, client=mock_client(handler), cache=cache, env={"EBIRD_API_KEY": KEY})
    assert (calls[0].url.params["dist"], calls[0].url.params["back"]) == ("50", "30")


def test_rejected_key_degrades_to_simulated(cache):
    handler, _ = serve(status=403, text="{}")
    out = ebird.get_recent_observations(40.64, -73.78, client=mock_client(handler), cache=cache, env={"EBIRD_API_KEY": KEY})
    assert (out["source"], out["fallback_reason"]) == ("simulated", "http_403")
    assert KEY not in json.dumps(out)


def test_malformed_answer_degrades_to_simulated(cache):
    handler, _ = serve(text='{"errors": []}')
    out = ebird.get_recent_observations(40.64, -73.78, client=mock_client(handler), cache=cache, env={"EBIRD_API_KEY": KEY})
    assert (out["source"], out["fallback_reason"]) == ("simulated", "bad_payload")


def test_empty_sightings_are_valid_live_data(cache):
    handler, _ = serve(text="[]")
    out = ebird.get_recent_observations(40.64, -73.78, client=mock_client(handler), cache=cache, env={"EBIRD_API_KEY": KEY})
    assert (out["source"], out["n_records"], out["individuals"], out["top_species"]) == ("live", 0, 0, [])


def test_ttl_is_24_hours(cache, clock):
    handler, calls = serve()
    client = mock_client(handler)
    env = {"EBIRD_API_KEY": KEY}
    ebird.get_recent_observations(40.64, -73.78, client=client, cache=cache, env=env)
    clock.advance(23 * 3600)
    assert ebird.get_recent_observations(40.64, -73.78, client=client, cache=cache, env=env)["source"] == "cached"
    clock.advance(2 * 3600)
    assert ebird.get_recent_observations(40.64, -73.78, client=client, cache=cache, env=env)["source"] == "live"
    assert len(calls) == 2
