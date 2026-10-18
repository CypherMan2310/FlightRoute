"""TTL cache, HTTP helper and the live / cached / simulated plumbing shared by every connector.

Contract for every connector payload: `source` is one of live | cached | simulated, plus `fetched_at`
(epoch s) and `age_s`. A simulated payload also carries `fallback_reason` (why the upstream was not used).
Multi-item payloads (METAR stations, forecast points) stamp each item and report the least-real one.
"""
from __future__ import annotations

import copy
import threading
import time
from dataclasses import dataclass, field
from typing import Any, Callable, Iterable

import httpx

TIMEOUT_S = 8.0
STALE_FACTOR = 3  # an expired entry may still be served as "cached" for up to ttl * STALE_FACTOR
USER_AGENT = "AEROGRID/0.1 (AERO-SIT college project; free, non-commercial)"
_RANK = {"live": 0, "cached": 1, "simulated": 2}
NEGATIVE = "__missing__"  # cache marker: the upstream answered without this id


class UpstreamUnavailable(Exception):
    """Raised by a connector when the upstream cannot be used; `reason` is a short machine-readable code."""

    def __init__(self, reason: str):
        super().__init__(reason)
        self.reason = reason


class NoSimulation(Exception):
    """Raised by a simulator that cannot fabricate a record for this id (e.g. a station not in the seed)."""


class TTLCache:
    def __init__(self, clock: Callable[[], float] = time.time, max_entries: int = 2048):
        self.clock = clock
        self.max_entries = max_entries
        self._items: dict[str, tuple[Any, float, float]] = {}
        self._lock = threading.Lock()

    def set(self, key: str, value: Any, ttl: float) -> None:
        with self._lock:
            self._items[key] = (value, self.clock(), ttl)
            if len(self._items) > self.max_entries:
                self._evict()

    def get(self, key: str, stale: bool = False) -> tuple[Any, float] | None:
        """(value, age_s) if fresh, or within ttl * STALE_FACTOR when `stale`; otherwise None."""
        with self._lock:
            item = self._items.get(key)
        if item is None:
            return None
        value, stored_at, ttl = item
        age = self.clock() - stored_at
        if age <= ttl * (STALE_FACTOR if stale else 1):
            return value, age
        return None

    def clear(self) -> None:
        with self._lock:
            self._items.clear()

    def __len__(self) -> int:
        return len(self._items)

    def _evict(self) -> None:
        now = self.clock()
        for k in [k for k, (_, t, ttl) in self._items.items() if now - t > ttl * STALE_FACTOR]:
            del self._items[k]
        while len(self._items) > self.max_entries:
            del self._items[min(self._items, key=lambda k: self._items[k][1])]


_default_cache: TTLCache | None = None


def default_cache() -> TTLCache:
    global _default_cache
    if _default_cache is None:
        _default_cache = TTLCache()
    return _default_cache


def _request(method: str, url: str, *, params=None, headers=None, data=None, client=None, timeout=TIMEOUT_S):
    hdrs = {"User-Agent": USER_AGENT, **(headers or {})}
    if client is not None:
        resp = client.request(method, url, params=params, headers=hdrs, data=data, timeout=timeout)
    else:
        with httpx.Client(timeout=timeout, follow_redirects=True) as c:
            resp = c.request(method, url, params=params, headers=hdrs, data=data)
    resp.raise_for_status()
    return resp


def http_get(url: str, *, params=None, headers=None, client=None, timeout: float = TIMEOUT_S) -> httpx.Response:
    return _request("GET", url, params=params, headers=headers, client=client, timeout=min(timeout, TIMEOUT_S))


def http_post(url: str, *, data=None, headers=None, client=None, timeout: float = TIMEOUT_S) -> httpx.Response:
    return _request("POST", url, data=data, headers=headers, client=client, timeout=min(timeout, TIMEOUT_S))


def describe_error(exc: BaseException) -> str:
    """Short reason code for `fallback_reason`. Never includes URLs, headers or exception text (no secrets)."""
    if isinstance(exc, UpstreamUnavailable):
        return exc.reason
    if isinstance(exc, httpx.TimeoutException):
        return "timeout"
    if isinstance(exc, httpx.HTTPStatusError):
        return f"http_{exc.response.status_code}"
    if isinstance(exc, httpx.TransportError):
        return "network_error"
    if isinstance(exc, (ValueError, KeyError, TypeError, IndexError, AttributeError, SyntaxError)):
        return "bad_payload"  # SyntaxError covers xml ParseError
    return type(exc).__name__


def combine_sources(sources: Iterable[str]) -> str:
    sources = list(sources)
    if not sources:
        return "simulated"
    return max(sources, key=_RANK.__getitem__)


def _stamp(payload: dict, source: str, fetched_at: float, now: float, **extra) -> dict:
    return {**payload, "source": source, "fetched_at": fetched_at, "age_s": max(0.0, now - fetched_at), "fallback_reason": None, **extra}


def fetch_one(cache: TTLCache, key: str, *, ttl: float, fetch: Callable[[], dict], simulate: Callable[[], dict]) -> dict:
    """fresh cache -> live fetch -> stale cache -> simulated. Never raises because of the upstream."""
    now = cache.clock()
    hit = cache.get(key)
    if hit:
        payload, _ = hit
        return _stamp(copy.deepcopy(payload), "cached", payload["fetched_at"], now)
    try:
        payload = {**fetch(), "fetched_at": cache.clock()}
    except Exception as exc:  # noqa: BLE001 - any upstream failure must degrade, not crash
        reason = describe_error(exc)
        stale = cache.get(key, stale=True)
        if stale:
            payload, _ = stale
            return _stamp(copy.deepcopy(payload), "cached", payload["fetched_at"], now, stale=True, fallback_reason=reason)
        return _stamp(simulate(), "simulated", now, now, fallback_reason=reason)
    cache.set(key, copy.deepcopy(payload), ttl)
    return _stamp(payload, "live", payload["fetched_at"], cache.clock())


@dataclass
class Batch:
    records: dict[str, dict] = field(default_factory=dict)
    missing: list[str] = field(default_factory=list)
    source: str = "simulated"
    fallback_reason: str | None = None

    def meta(self, provider: str) -> dict:
        """Top-level payload fields: least-real source, oldest fetch, largest age."""
        recs = self.records.values()
        return {
            "source": self.source,
            "provider": provider,
            "fetched_at": min((r["fetched_at"] for r in recs), default=None),
            "age_s": max((r["age_s"] for r in recs), default=0.0),
            "fallback_reason": self.fallback_reason,
        }


def batch_fetch(
    cache: TTLCache,
    namespace: str,
    ids: list[str],
    *,
    ttl: float,
    fetch_many: Callable[[list[str]], dict[str, dict]],
    simulate_one: Callable[[str], dict],
) -> Batch:
    """Per-item cache. Only uncached ids go upstream. Ids the upstream answered without are `missing`
    (not simulated); a failed or empty answer falls back per id to stale cache, then simulation."""
    now = cache.clock()
    out: dict[str, dict] = {}
    todo: list[str] = []
    missing: list[str] = []
    for i in ids:
        hit = cache.get(f"{namespace}:{i}")
        if hit and hit[0].get(NEGATIVE):
            missing.append(i)  # the upstream recently answered without this id: do not ask again until the TTL expires
        elif hit:
            rec, _ = hit
            out[i] = _stamp(copy.deepcopy(rec), "cached", rec["fetched_at"], now)
        else:
            todo.append(i)
    reason: str | None = None
    if todo:
        try:
            got = fetch_many(todo)
            if not got:
                raise UpstreamUnavailable("empty_response")
            t = cache.clock()
            for i in todo:
                if i in got:
                    rec = {**got[i], "fetched_at": t}
                    cache.set(f"{namespace}:{i}", copy.deepcopy(rec), ttl)
                    out[i] = _stamp(rec, "live", t, t)
                else:
                    cache.set(f"{namespace}:{i}", {NEGATIVE: True, "fetched_at": t}, ttl)
                    missing.append(i)
        except Exception as exc:  # noqa: BLE001
            reason = describe_error(exc)
            for i in todo:
                stale = cache.get(f"{namespace}:{i}", stale=True)
                if stale and stale[0].get(NEGATIVE):
                    missing.append(i)
                elif stale:
                    rec, _ = stale
                    out[i] = _stamp(copy.deepcopy(rec), "cached", rec["fetched_at"], now, stale=True, fallback_reason=reason)
                else:
                    try:
                        out[i] = _stamp(simulate_one(i), "simulated", now, now, fallback_reason=reason)
                    except NoSimulation:
                        missing.append(i)
    records = {i: out[i] for i in ids if i in out}
    source = combine_sources(r["source"] for r in records.values()) if records or reason else "cached"  # only known-missing ids: answered from the cache
    return Batch(records, [i for i in ids if i in set(missing)], source, reason)
