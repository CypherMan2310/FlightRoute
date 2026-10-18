import httpx
import pytest

from aerogrid.sources.cache import (
    STALE_FACTOR,
    TIMEOUT_S,
    NoSimulation,
    TTLCache,
    UpstreamUnavailable,
    batch_fetch,
    combine_sources,
    describe_error,
    fetch_one,
    http_get,
)

from conftest import mock_client


def test_timeout_is_at_most_eight_seconds():
    assert 0 < TIMEOUT_S <= 8.0


def test_ttl_cache_expires(clock):
    c = TTLCache(clock=clock)
    c.set("k", {"v": 1}, ttl=10)
    assert c.get("k")[0] == {"v": 1}
    clock.advance(10.5)
    assert c.get("k") is None
    assert c.get("k", stale=True)[0] == {"v": 1}
    clock.advance(19)  # age 29.5 s = just inside 3 x ttl
    assert c.get("k", stale=True)[0] == {"v": 1}
    clock.advance(1)  # age 30.5 s = just outside
    assert c.get("k", stale=True) is None
    assert STALE_FACTOR == 3


def test_ttl_cache_is_bounded(clock):
    c = TTLCache(clock=clock, max_entries=3)
    for i in range(5):
        c.set(f"k{i}", i, ttl=100)
        clock.advance(1)
    assert len(c) == 3
    assert c.get("k4") is not None and c.get("k0") is None


def test_combine_sources_is_pessimistic():
    assert combine_sources(["live", "live"]) == "live"
    assert combine_sources(["live", "cached"]) == "cached"
    assert combine_sources(["live", "cached", "simulated"]) == "simulated"
    assert combine_sources([]) == "simulated"


@pytest.mark.parametrize(
    "exc,reason",
    [
        (httpx.ReadTimeout("x"), "timeout"),
        (httpx.ConnectError("x"), "network_error"),
        (httpx.HTTPStatusError("x", request=httpx.Request("GET", "http://a"), response=httpx.Response(503)), "http_503"),
        (ValueError("x"), "bad_payload"),
        (UpstreamUnavailable("no_api_key"), "no_api_key"),
    ],
)
def test_describe_error(exc, reason):
    assert describe_error(exc) == reason


def test_http_get_raises_on_error_status_and_sends_user_agent():
    seen = {}

    def handler(request):
        seen["ua"] = request.headers["user-agent"]
        return httpx.Response(500)

    with pytest.raises(httpx.HTTPStatusError):
        http_get("https://example.test/x", client=mock_client(handler))
    assert "AEROGRID" in seen["ua"]


def test_fetch_one_live_then_cached_then_live_again(cache, clock):
    calls = []

    def fetch():
        calls.append(1)
        return {"n": len(calls)}

    a = fetch_one(cache, "k", ttl=60, fetch=fetch, simulate=lambda: {"n": -1})
    assert (a["source"], a["n"], a["age_s"]) == ("live", 1, 0)
    clock.advance(30)
    b = fetch_one(cache, "k", ttl=60, fetch=fetch, simulate=lambda: {"n": -1})
    assert (b["source"], b["n"], b["age_s"]) == ("cached", 1, 30)
    assert len(calls) == 1
    clock.advance(31)
    c = fetch_one(cache, "k", ttl=60, fetch=fetch, simulate=lambda: {"n": -1})
    assert (c["source"], c["n"]) == ("live", 2)


def test_every_payload_carries_the_contract_fields(cache, clock):
    for _ in range(2):  # live, then cached
        p = fetch_one(cache, "k", ttl=60, fetch=lambda: {"n": 1}, simulate=lambda: {"n": -1})
        assert {"source", "fetched_at", "age_s", "fallback_reason"} <= set(p)
        assert p["fallback_reason"] is None


def test_fetch_one_falls_back_to_stale_cache_then_simulated(cache, clock):
    fetch_one(cache, "k", ttl=60, fetch=lambda: {"n": 1}, simulate=lambda: {"n": -1})
    clock.advance(100)

    def boom():
        raise httpx.ReadTimeout("slow")

    stale = fetch_one(cache, "k", ttl=60, fetch=boom, simulate=lambda: {"n": -1})
    assert (stale["source"], stale["n"], stale["stale"], stale["fallback_reason"]) == ("cached", 1, True, "timeout")
    clock.advance(60 * STALE_FACTOR)
    sim = fetch_one(cache, "k", ttl=60, fetch=boom, simulate=lambda: {"n": -1})
    assert (sim["source"], sim["n"], sim["fallback_reason"]) == ("simulated", -1, "timeout")


def test_fetch_one_converts_any_fetch_exception_to_simulated(cache):
    out = fetch_one(cache, "k", ttl=1, fetch=lambda: 1 / 0, simulate=lambda: {})
    assert (out["source"], out["fallback_reason"]) == ("simulated", "ZeroDivisionError")


def test_batch_fetch_mixes_cached_and_live(cache, clock):
    fetched = []

    def fetch_many(ids):
        fetched.append(list(ids))
        return {i: {"v": i.lower()} for i in ids}

    first = batch_fetch(cache, "m", ["A", "B"], ttl=60, fetch_many=fetch_many, simulate_one=lambda i: {"v": "sim"})
    assert {r["source"] for r in first.records.values()} == {"live"}
    second = batch_fetch(cache, "m", ["A", "B", "C"], ttl=60, fetch_many=fetch_many, simulate_one=lambda i: {"v": "sim"})
    assert fetched == [["A", "B"], ["C"]]
    assert [second.records[i]["source"] for i in "ABC"] == ["cached", "cached", "live"]
    assert second.source == "cached"
    assert second.missing == []


def test_batch_fetch_reports_missing_ids_without_simulating_them(cache):
    out = batch_fetch(cache, "m", ["A", "B"], ttl=60, fetch_many=lambda ids: {"A": {"v": 1}}, simulate_one=lambda i: {"v": 0})
    assert list(out.records) == ["A"]
    assert out.missing == ["B"]


def test_batch_fetch_marks_unsimulatable_ids_missing(cache):
    def boom(ids):
        raise httpx.ConnectError("down")

    def sim(i):
        if i == "B":
            raise NoSimulation(i)
        return {"v": i}

    out = batch_fetch(cache, "m", ["A", "B"], ttl=60, fetch_many=boom, simulate_one=sim)
    assert list(out.records) == ["A"] and out.missing == ["B"]


def test_batch_fetch_simulates_everything_on_total_failure(cache):
    def boom(ids):
        raise httpx.ConnectError("down")

    out = batch_fetch(cache, "m", ["A", "B"], ttl=60, fetch_many=boom, simulate_one=lambda i: {"v": i})
    assert out.source == "simulated"
    assert {r["source"] for r in out.records.values()} == {"simulated"}
    assert out.fallback_reason == "network_error"


def test_batch_fetch_treats_empty_upstream_answer_as_failure(cache):
    out = batch_fetch(cache, "m", ["A"], ttl=60, fetch_many=lambda ids: {}, simulate_one=lambda i: {"v": i})
    assert out.source == "simulated"
    assert out.fallback_reason == "empty_response"


def test_batch_fetch_negative_caches_ids_the_upstream_answered_without(cache, clock):
    calls = []

    def fetch_many(ids):
        calls.append(list(ids))
        return {"A": {"v": 1}} if "A" in ids else {}

    first = batch_fetch(cache, "m", ["A", "B"], ttl=60, fetch_many=fetch_many, simulate_one=lambda i: {"v": 0})
    assert first.missing == ["B"] and calls == [["A", "B"]]
    again = batch_fetch(cache, "m", ["A", "B"], ttl=60, fetch_many=fetch_many, simulate_one=lambda i: {"v": 0})
    assert calls == [["A", "B"]], "nothing goes upstream: A is cached and B is known to have no data"
    assert again.missing == ["B"] and again.source == "cached" and list(again.records) == ["A"]
    clock.advance(61)
    batch_fetch(cache, "m", ["A", "B"], ttl=60, fetch_many=fetch_many, simulate_one=lambda i: {"v": 0})
    assert len(calls) == 2, "the negative entry expires with the TTL"


def test_an_id_with_no_data_alone_in_a_request_does_not_turn_into_a_simulation(cache, clock):
    def answered_without_it(ids):
        return {"A": {"v": 1}} if "A" in ids else {}

    batch_fetch(cache, "m", ["A", "B"], ttl=60, fetch_many=answered_without_it, simulate_one=lambda i: {"v": 0})
    clock.advance(30)
    out = batch_fetch(cache, "m", ["A", "B"], ttl=60, fetch_many=answered_without_it, simulate_one=lambda i: {"v": 0})
    assert out.source == "cached" and out.fallback_reason is None
