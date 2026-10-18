import json
import time

import httpx
import pandas as pd
import pytest
from fastapi.testclient import TestClient

from aerogrid import data
from aerogrid.api.main import Sources, create_app
from aerogrid.sources.cache import TTLCache
from aerogrid.train import synth, train_all

from conftest import FakeClock, fixture_json, fixture_text, mock_client


@pytest.fixture(scope="module")
def models_dir(tmp_path_factory):
    out = tmp_path_factory.mktemp("api-models")
    cfg = train_all.Config(n_folds=2, gcn_hidden=16, gcn_epochs=30, gcn_lr=5e-3, rounds=80, bootstrap=20)
    train_all.train(synth.make_synthetic(n_days=30, n_airports=10, seed=5, flights_scale=1.0), out, config=cfg)
    return out


def all_down(request):
    return httpx.Response(503)


def all_up(clock):
    """Serves every upstream from recorded fixtures, scaled to however many stations / points are asked for."""

    def handler(request: httpx.Request) -> httpx.Response:
        host, path, q = request.url.host, request.url.path, request.url.params
        if host == "aviationweather.gov" and path.endswith("/metar"):
            rows = []
            for icao in q["ids"].split(","):
                row = dict(fixture_json("metar_kjfk_kord_vidp.json")[0])
                row["icaoId"] = icao
                rows.append(row)
            return httpx.Response(200, json=rows)
        if host == "aviationweather.gov" and path.endswith("/taf"):
            return httpx.Response(200, text=fixture_text("taf_kjfk.json"), headers={"content-type": "application/json"})
        if host == "api.open-meteo.com":
            n = len(q["latitude"].split(","))
            hours = int(q["forecast_hours"])
            t0 = int(clock() // 3600) * 3600
            times = pd.to_datetime([t0 + 3600 * h for h in range(hours)], unit="s").strftime("%Y-%m-%dT%H:%M").tolist()
            one = {"hourly": {"time": times, "temperature_2m": [15.0] * hours, "precipitation": [0.0] * hours, "wind_speed_10m": [8.0] * hours, "wind_gusts_10m": [15.0] * hours,
                              "visibility": [24000.0] * hours, "cloud_cover_low": [10] * hours, "cape": [0.0] * hours, "weather_code": [1] * hours,
                              "wind_speed_250hPa": [90.0] * hours, "wind_direction_250hPa": [270] * hours}}
            return httpx.Response(200, json=[one] * n)
        if host == "nasstatus.faa.gov":
            return httpx.Response(200, text=fixture_text("nas_status.xml"), headers={"content-type": "application/xml"})
        if host == "api.adsb.lol":
            return httpx.Response(200, text=fixture_text("adsblol_point.json"), headers={"content-type": "application/json"})
        return httpx.Response(404)

    return handler


def om_429_metno_up(clock):
    """Every upstream is up except Open-Meteo, which answers 429 (what Render's shared IP got); MET Norway serves the recorded answer."""
    base = all_up(clock)

    def handler(request: httpx.Request) -> httpx.Response:
        if request.url.host == "api.open-meteo.com":
            return httpx.Response(429)
        if request.url.host == "api.met.no":
            return httpx.Response(200, json=fixture_json("metno_complete_ord.json"))
        return base(request)

    return handler


@pytest.fixture(autouse=True)
def no_throttle(monkeypatch):
    monkeypatch.setattr("aerogrid.sources.metno.MIN_INTERVAL_S", 0.0)


def make_client(models_dir, tmp_path, handler=all_down, clock=None, models=True, **kw):
    clock = clock or FakeClock()
    sources = Sources(client=mock_client(handler(clock) if handler is all_up else handler), cache=TTLCache(clock=clock), env={})
    app = create_app(models_dir=models_dir if models else tmp_path / "no-models", db_path=tmp_path / "score.db", sources=sources, start_background=False, clock=clock, **kw)
    return TestClient(app), app, clock


@pytest.fixture
def down(models_dir, tmp_path):
    return make_client(models_dir, tmp_path)


@pytest.fixture
def up(models_dir, tmp_path):
    clock = FakeClock()
    return make_client(models_dir, tmp_path, handler=all_up, clock=clock)


def test_health_reports_models_and_never_needs_the_network(down):
    client, _, _ = down
    r = client.get("/api/health")
    j = r.json()
    assert r.status_code == 200 and j["status"] == "ok" and j["models"]["status"] == "trained" and j["models"]["trained_on"] == "synthetic"
    assert j["version"]


def test_models_endpoint_returns_the_card_and_data_source_list(down):
    client, _, _ = down
    j = client.get("/api/models").json()
    assert j["trained"] is True and j["card"]["models"]["ensemble"]["kind"] == "stacked"
    names = {s["name"] for s in j["data_sources"]}
    assert {"aviationweather.gov", "Open-Meteo", "adsb.lol", "FAA NAS Status", "eBird"} <= names
    assert all({"ttl_s", "license", "role"} <= set(s) for s in j["data_sources"])
    assert j["badges"]["routing"] == "algorithmic (not ML)" and j["badges"]["bird"] == "prior-based estimate"


def test_airports_search(down):
    client, _, _ = down
    j = client.get("/api/airports", params={"q": "jfk"}).json()
    assert j[0]["iata"] == "JFK" and j[0]["coverage"] == "us"
    assert {"iata", "icao", "name", "city", "country", "lat", "lon", "tier"} <= set(j[0])
    assert any(a["iata"] == "LHR" and a["coverage"] == "world (lower confidence)" for a in client.get("/api/airports", params={"q": "london"}).json())
    assert client.get("/api/airports", params={"q": "zzzzzz"}).json() == []
    assert len(client.get("/api/airports", params={"limit": 5}).json()) == 5
    assert client.get("/api/airports", params={"limit": 0}).status_code == 422


def test_flights_degrade_to_simulated_when_every_upstream_is_down(down):
    client, _, _ = down
    j = client.get("/api/flights", params={"lat": 40.64, "lon": -73.78, "radius": 100}).json()
    assert j["source"] == "simulated" and j["fallback_reason"] and j["count"] == len(j["flights"])
    assert client.get("/api/flights", params={"lat": 95, "lon": 0}).status_code == 422


def test_flights_live_path_uses_adsb_lol(up):
    client, _, _ = up
    j = client.get("/api/flights", params={"lat": 40.64, "lon": -73.78}).json()
    assert j["source"] == "live" and j["provider"] == "adsb.lol" and j["count"] == 7


def test_global_flights_merge_regions_and_report_the_least_real_source(down):
    client, _, _ = down
    j = client.get("/api/flights").json()
    assert j["source"] == "simulated" and j["count"] > 0 and j["regions"] >= 1


def test_congestion_ranks_airports_by_stress_with_confidence_and_provenance(down):
    client, _, _ = down
    j = client.get("/api/congestion", params={"limit": 20}).json()
    assert j["trained_on"] == "synthetic" and j["source"] == "simulated" and len(j["airports"]) == 20
    stress = [a["stress"] for a in j["airports"]]
    assert stress == sorted(stress, reverse=True) and all(0 <= s <= 100 for s in stress)
    a = j["airports"][0]
    assert {"iata", "name", "lat", "lon", "stress", "pred_dep_delay_min", "pred_nas_min", "weather_category", "nas", "confidence", "tier"} <= set(a)
    assert {x["confidence"] for x in j["airports"]} <= {"normal", "low"}
    assert client.get("/api/congestion", params={"limit": 1000}).status_code == 422


def test_congestion_live_path(up):
    client, _, _ = up
    j = client.get("/api/congestion").json()
    assert j["source"] in {"live", "cached"} and j["sources"]["metar"]["source"] == "live"
    bwi = next((a for a in client.get("/api/congestion", params={"limit": 155}).json()["airports"] if a["iata"] == "BWI"), None)
    assert bwi["nas"]["status"] == "delay" and bwi["nas"]["delay_minutes"] == 90


def predict(client, **body):
    body = {"origin": "JFK", "destination": "LAX", "airline": "AA", **body}
    return client.post("/api/predict", json=body)


def test_predict_returns_additive_causes_interval_drivers_and_badges(down):
    client, _, _ = down
    r = predict(client)
    assert r.status_code == 200
    j = r.json()
    assert j["origin"] == "JFK" and j["trained_on"] == "synthetic" and j["horizon_h"] in {2, 6, 24}
    assert [c["key"] for c in j["causes"]].count("weather") == 1 and {c["key"] for c in j["causes"]} == {"weather", "congestion", "cascade", "route", "bird", "carrier"}
    assert sum(c["minutes"] for c in j["causes"]) + j["baseline_min"] == pytest.approx(j["delay_min"], abs=0.1)
    assert all(c["minutes"] >= 0 for c in j["causes"]) and j["baseline_min"] >= 0
    lo, hi = j["interval80"]
    assert 0 <= lo <= hi and hi >= j["delay_min"] - 1e-6
    assert set(j["drivers"]) == {"weather", "congestion", "cascade", "route", "bird", "carrier"}
    assert j["drivers"]["weather"] and {"feature", "label", "contribution_min"} <= set(j["drivers"]["weather"][0])
    assert j["models"]["routing"]["label"] == "algorithmic (not ML)" and j["models"]["bird"]["label"] == "prior-based estimate" and j["models"]["weather"]["trained_on"] == "synthetic"
    assert j["sources"]["metar"]["source"] == "simulated" and j["sources"]["forecast"]["source"] == "simulated"
    assert j["confidence"]["level"] == "normal" and any("synthetic" in n.lower() for n in j["confidence"]["notes"])
    assert j["route"]["label"] == "algorithmic (not ML)" and j["bird"]["origin"]["label"] == "prior-based estimate"
    assert j["advice"] and all(isinstance(a, str) for a in j["advice"])


def test_predict_live_sources_and_nas_status_shown_beside_the_prediction(up):
    client, _, _ = up
    j = predict(client, origin="BWI", destination="ATL", airline="DL").json()
    assert j["sources"]["metar"]["source"] == "live" and j["sources"]["forecast"]["source"] == "live" and j["sources"]["nas"]["source"] == "live"
    assert j["nas"]["origin"]["status"] == "delay" and j["nas"]["origin"]["delay_minutes"] == 90 and j["nas"]["destination"]["status"] == "normal"


def test_predict_falls_back_to_met_norway_and_says_what_that_costs(models_dir, tmp_path):
    clock = FakeClock()
    client, _, _ = make_client(models_dir, tmp_path, handler=om_429_metno_up(clock), clock=clock)
    j = predict(client, origin="BWI", destination="ATL", airline="DL").json()
    f, e = j["sources"]["forecast"], j["sources"]["enroute_forecast"]
    assert (f["source"], f["provider"], f["fallback_reason"]) == ("live", "api.met.no", "http_429") and e["provider"] == "api.met.no"
    assert set(f["degraded"]) == {"gust_kt", "visibility_m", "cape_jkg", "wind250"} and "MET Norway" in f["attribution"]
    assert j["sources"]["metar"]["source"] == "live" and j["delay_min"] is not None
    assert j["confidence"]["level"] == "low" and any("MET Norway" in n and "gust" in n for n in j["confidence"]["notes"])
    assert "unavailable" in j["drivers"]["route"][0]["note"]


def test_advisor_says_when_jet_level_wind_is_missing(models_dir, tmp_path):
    clock = FakeClock()
    client, _, _ = make_client(models_dir, tmp_path, handler=om_429_metno_up(clock), clock=clock)
    j = client.get("/api/advisor", params={"origin": "JFK", "destination": "LAX"}).json()
    assert j["sources"]["enroute_forecast"]["provider"] == "api.met.no" and any("Jet-level wind is unavailable" in a for a in j["advice"])


def test_models_lists_met_norway_as_the_forecast_fallback_and_shows_when_it_was_used(models_dir, tmp_path):
    clock = FakeClock()
    client, _, _ = make_client(models_dir, tmp_path, handler=om_429_metno_up(clock), clock=clock)
    predict(client, origin="BWI", destination="ATL", airline="DL")
    rows = {s["name"]: s for s in client.get("/api/models").json()["data_sources"]}
    assert rows["MET Norway"]["last"]["provider"] == "api.met.no" and rows["MET Norway"]["last"]["degraded"]
    assert rows["Open-Meteo"]["last"]["provider"] == "api.met.no", "the forecast row reports what the models actually received"


def test_tick_does_not_score_the_app_on_the_reduced_met_norway_forecast(models_dir, tmp_path):
    clock = FakeClock()
    _, app, _ = make_client(models_dir, tmp_path, handler=om_429_metno_up(clock), clock=clock)
    assert app.state.tick()["logged"] == 0


def test_outside_the_us_confidence_is_low(down):
    client, _, _ = down
    j = predict(client, origin="LHR", destination="CDG", airline="BA").json()
    assert j["confidence"]["level"] == "low" and any("US" in n for n in j["confidence"]["notes"])


def test_predict_accepts_icao_codes_and_an_explicit_departure_time(down):
    client, _, clock = down
    dep = pd.Timestamp(clock() + 5 * 3600, unit="s", tz="UTC").isoformat()
    j = predict(client, origin="KJFK", destination="KORD", departure_time=dep).json()
    assert j["origin"] == "JFK" and j["destination"] == "ORD" and j["horizon_h"] == 6


@pytest.mark.parametrize(
    "body,status",
    [({"origin": "ZZZ"}, 404), ({"destination": "QQQ"}, 404), ({"origin": "JFK", "destination": "JFK"}, 422), ({"origin": "J"}, 422), ({"aircraft": "balloon"}, 422)],
)
def test_predict_rejects_bad_requests(down, body, status):
    client, _, _ = down
    assert predict(client, **body).status_code == status


def test_predict_rejects_departures_in_the_past_or_beyond_the_forecast_window(down):
    client, _, clock = down
    past = pd.Timestamp(clock() - 6 * 3600, unit="s", tz="UTC").isoformat()
    far = pd.Timestamp(clock() + 60 * 3600, unit="s", tz="UTC").isoformat()
    assert predict(client, departure_time=past).status_code == 422
    assert predict(client, departure_time=far).status_code == 422


def test_advisor_returns_route_plan_bird_bands_and_wind_effect(down):
    client, _, _ = down
    j = client.get("/api/advisor", params={"origin": "JFK", "destination": "LHR", "aircraft": "narrowbody"}).json()
    assert j["route"]["feasible"] and j["route"]["stops"] and j["route"]["label"] == "algorithmic (not ML)"
    assert j["bird"]["origin"]["altitude_band_ft"][1] > j["bird"]["origin"]["altitude_band_ft"][0]
    assert {"delta_min", "extra_min"} <= set(j["wind"]) and j["advice"]
    assert client.get("/api/advisor", params={"origin": "JFK", "destination": "ZZZ"}).status_code == 404


def test_whatif_cascade_through_the_graph(down):
    client, _, _ = down
    r = client.post("/api/whatif", json={"airport": "ATL", "kind": "storm", "severity": 1.0})
    assert r.status_code == 200
    j = r.json()
    assert j["label"] == "model-based scenario (not a forecast)" and j["trained_on"] == "synthetic" and j["airport"] == "ATL"
    assert j["affected"] and j["affected"][0]["hops"] is not None and len(j["deltas"]) == 155
    assert client.post("/api/whatif", json={"airport": "ZZZ", "kind": "storm", "severity": 0.5}).status_code == 404
    assert client.post("/api/whatif", json={"airport": "ATL", "kind": "meteor", "severity": 0.5}).status_code == 422
    assert client.post("/api/whatif", json={"airport": "ATL", "kind": "storm", "severity": 2}).status_code == 422


def test_scorecard_endpoint_reports_ground_truth_and_empty_state(down):
    client, _, _ = down
    j = client.get("/api/scorecard").json()
    assert j["n_resolved"] == 0 and j["insufficient_data"] is True and "FAA NAS Status" in j["ground_truth"]
    assert j["trained_on"] == "synthetic" and j["scored_airports"] > 0


def test_tick_logs_predictions_for_us_airports_and_resolves_them_against_live_nas(models_dir, tmp_path):
    clock = FakeClock()
    client, app, _ = make_client(models_dir, tmp_path, handler=all_up, clock=clock)
    logged = app.state.tick()
    assert logged["logged"] > 0 and logged["resolved"] == 0
    clock.advance(2 * 3600 + 60)
    after = app.state.tick()
    assert after["resolved"] == logged["logged"]
    m = client.get("/api/scorecard").json()
    assert m["n_resolved"] == logged["logged"] and m["confusion"]["tp"] + m["confusion"]["fp"] + m["confusion"]["fn"] + m["confusion"]["tn"] == m["n_resolved"]


def test_tick_never_logs_predictions_made_on_simulated_weather_nor_resolves_on_simulated_nas(models_dir, tmp_path):
    clock = FakeClock()
    client, app, _ = make_client(models_dir, tmp_path, handler=all_down, clock=clock)
    first = app.state.tick()
    assert first["logged"] == 0 and first["resolved"] == 0, "METAR / forecast are simulated, so nothing may enter the app's accuracy record"
    clock.advance(2 * 3600 + 60)
    assert app.state.tick()["resolved"] == 0
    assert client.get("/api/scorecard").json()["n_resolved"] == 0


def test_stream_emits_server_sent_events(down):
    client, _, _ = down
    with client.stream("GET", "/api/stream", params={"max_events": 2, "interval_s": 0}) as r:
        assert r.headers["content-type"].startswith("text/event-stream")
        body = "".join(r.iter_text())
    events = [line.split(": ", 1)[1] for line in body.splitlines() if line.startswith("event: ")]
    assert events[:2] == ["flights", "congestion"]
    first = next(line for line in body.splitlines() if line.startswith("data: "))
    assert "source" in json.loads(first[6:])


def test_without_trained_models_model_endpoints_say_so_but_the_rest_works(models_dir, tmp_path):
    client, _, _ = make_client(models_dir, tmp_path, models=False)
    health = client.get("/api/health").json()["models"]
    assert health["status"] == "untrained" and "model_card.json" in health["reason"]
    assert client.get("/api/models").json()["trained"] is False
    assert client.get("/api/airports", params={"q": "jfk"}).status_code == 200
    assert client.get("/api/flights", params={"lat": 40, "lon": -73}).status_code == 200
    for call in (lambda: predict(client), lambda: client.get("/api/congestion"), lambda: client.post("/api/whatif", json={"airport": "ATL", "kind": "storm", "severity": 0.5})):
        r = call()
        assert r.status_code == 503 and "not trained" in r.json()["detail"].lower()


def test_cors_allows_the_dev_frontend(down):
    client, _, _ = down
    r = client.options("/api/health", headers={"Origin": "http://localhost:5173", "Access-Control-Request-Method": "GET"})
    assert r.headers.get("access-control-allow-origin") in {"*", "http://localhost:5173"}
