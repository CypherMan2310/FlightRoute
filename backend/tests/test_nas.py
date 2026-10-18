from datetime import datetime, timezone

import httpx
import pytest

from aerogrid.sources import nas

from conftest import fixture_text, mock_client


def serve(text, status=200):
    calls = []

    def handler(request):
        calls.append(request)
        return httpx.Response(status, text=text, headers={"content-type": "application/xml"})

    return handler, calls


@pytest.mark.parametrize(
    "text,minutes",
    [("1 hour and 16 minutes", 76), ("45 minutes", 45), ("2 hours", 120), ("1 hour", 60), ("2 hours and 11 minutes", 131), ("", None), ("soon", None), (None, None)],
)
def test_parse_minutes(text, minutes):
    assert nas.parse_minutes(text) == minutes


def test_recorded_feed_is_normalised(cache):
    handler, calls = serve(fixture_text("nas_status.xml"))
    out = nas.get_status(client=mock_client(handler), cache=cache)
    assert (out["source"], out["provider"]) == ("live", "nasstatus.faa.gov")
    assert calls[0].url.host == "nasstatus.faa.gov" and calls[0].url.path == "/api/airport-status-information"
    assert out["updated"] == int(datetime(2026, 10, 17, 7, 16, 13, tzinfo=timezone.utc).timestamp())
    assert set(out["airports"]) == {"BWI", "MCO"}
    bwi = out["airports"]["BWI"]
    assert (bwi["status"], bwi["delay_minutes"], bwi["ground_stop"], bwi["ground_delay"]) == ("delay", 90, False, False)
    assert bwi["programs"] == [
        {"type": "general_delay", "direction": "departure", "reason": "WX:Thunderstorms", "min_minutes": 76, "max_minutes": 90, "trend": "Increasing"}
    ]
    assert out["airports"]["MCO"]["delay_minutes"] == 30


def test_closure_notams_are_not_treated_as_delay_ground_truth(cache):
    out = nas.get_status(client=mock_client(serve(fixture_text("nas_status.xml"))[0]), cache=cache)
    assert {n["airport"] for n in out["notams"]} == {"JAC", "SNA", "LAX", "SAN", "ASE"}
    assert not {"JAC", "SNA", "LAX", "SAN", "ASE"} & set(out["airports"])
    jac = next(n for n in out["notams"] if n["airport"] == "JAC")
    assert jac["start"] == "Oct 17 at 03:54 UTC." and "CLSD" in jac["reason"]


def test_ground_programs_take_precedence_over_plain_delays(cache):
    out = nas.get_status(client=mock_client(serve(fixture_text("nas_programs.constructed.xml"))[0]), cache=cache)
    ewr, den, ord_ = out["airports"]["EWR"], out["airports"]["DEN"], out["airports"]["ORD"]
    assert (ewr["status"], ewr["ground_delay"], ewr["delay_minutes"]) == ("ground_delay", True, 131)
    assert sorted(p["type"] for p in ewr["programs"]) == ["general_delay", "ground_delay"]
    gdp = next(p for p in ewr["programs"] if p["type"] == "ground_delay")
    assert (gdp["avg_minutes"], gdp["max_minutes"], gdp["reason"]) == (53, 131, "thunderstorms")
    assert (den["status"], den["ground_stop"], den["ground_delay"], den["delay_minutes"]) == ("ground_stop", True, True, 95)
    assert next(p for p in den["programs"] if p["type"] == "ground_stop")["end_time"] == "3:00 pm EDT"
    assert (ord_["status"], ord_["programs"][0]["direction"], ord_["delay_minutes"]) == ("delay", "arrival", 150)


def test_quiet_day_is_live_with_no_airports(cache):
    quiet = "<AIRPORT_STATUS_INFORMATION><Update_Time>Sat Oct 17 07:16:13 2026 GMT</Update_Time></AIRPORT_STATUS_INFORMATION>"
    out = nas.get_status(client=mock_client(serve(quiet)[0]), cache=cache)
    assert (out["source"], out["airports"], out["notams"]) == ("live", {}, [])


def test_airport_status_defaults_to_normal():
    payload = {"airports": {"BWI": {"status": "delay"}}}
    assert nas.airport_status(payload, "bwi") == {"status": "delay"}
    assert nas.airport_status(payload, "ATL") == {"status": "normal", "delay_minutes": 0, "ground_stop": False, "ground_delay": False, "programs": []}


def test_ttl_is_two_minutes(cache, clock):
    handler, calls = serve(fixture_text("nas_status.xml"))
    client = mock_client(handler)
    nas.get_status(client=client, cache=cache)
    clock.advance(119)
    assert nas.get_status(client=client, cache=cache)["source"] == "cached"
    clock.advance(2)
    assert nas.get_status(client=client, cache=cache)["source"] == "live"
    assert len(calls) == 2


@pytest.mark.parametrize(
    "response",
    [
        httpx.Response(503),
        httpx.Response(200, text="<html><body>maintenance</body></html>"),
        httpx.Response(200, text="<<not xml"),
        httpx.Response(200, text='<!DOCTYPE x [<!ENTITY a "b">]><AIRPORT_STATUS_INFORMATION>&a;</AIRPORT_STATUS_INFORMATION>'),
    ],
)
def test_bad_upstream_degrades_to_simulated(cache, response):
    out = nas.get_status(client=mock_client(lambda r: response), cache=cache)
    assert out["source"] == "simulated" and out["fallback_reason"]
    assert isinstance(out["airports"], dict) and isinstance(out["notams"], list)
