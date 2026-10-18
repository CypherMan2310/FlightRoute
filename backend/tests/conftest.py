"""Shared test helpers. Tests never touch the network: any real HTTP attempt fails the test."""
import json
from pathlib import Path

import httpx
import pytest

FIXTURES = Path(__file__).parent / "fixtures"


class FakeClock:
    """Injectable clock so TTL / age / simulated-time behaviour is deterministic."""

    def __init__(self, t: float = 1_792_221_414.0):
        self.t = t

    def __call__(self) -> float:
        return self.t

    def advance(self, seconds: float) -> None:
        self.t += seconds


@pytest.fixture
def clock():
    return FakeClock()


@pytest.fixture
def cache(clock):
    from aerogrid.sources.cache import TTLCache

    return TTLCache(clock=clock)


@pytest.fixture(autouse=True)
def forbid_real_network(monkeypatch):
    attempts = []

    def _blocked(self, request):
        attempts.append(str(request.url))
        raise httpx.ConnectError("network disabled in tests")

    monkeypatch.setattr(httpx.HTTPTransport, "handle_request", _blocked)
    yield
    assert not attempts, f"test attempted real network calls: {attempts}"


def fixture_text(name: str) -> str:
    return (FIXTURES / name).read_text(encoding="utf-8")


def fixture_json(name: str):
    return json.loads(fixture_text(name))


def mock_client(handler) -> httpx.Client:
    """httpx client whose transport is `handler(request) -> httpx.Response`."""
    return httpx.Client(transport=httpx.MockTransport(handler))


def json_response(name: str, status: int = 200) -> httpx.Response:
    return httpx.Response(status, text=fixture_text(name), headers={"content-type": "application/json"})
