"""MET Norway Locationforecast 2.0 (api.met.no, "complete" product): the keyless fallback when Open-Meteo is unavailable.

Used only for the points Open-Meteo could not serve (HTTP 429, daily budget, outage) and that have no cached Open-Meteo record
either, i.e. the points that would otherwise be simulated. A MET Norway record is real, but it is NOT equivalent: the product has
no wind gusts, visibility, CAPE or jet-level (250 hPa) wind, which the models were trained with. Those four are filled with
neutral stand-ins and listed in `degraded` on every record and on the payload, so the API, the UI and the scorecard can see it:
  gust_kt       -> sustained wind speed (a lower bound)
  visibility_m  -> 10 km (neutral good visibility)
  cape_jkg      -> 0
  wind250       -> 0 kt (no jet-level wind effect on the route)
Terms (checked 2026-10-18): free, no key; a User-Agent that identifies the application with contact information is
mandatory (set `AEROGRID_CONTACT`, default the project's repository URL); keep requests under about 20 per second, cache, and
credit "MET Norway" (CC BY 4.0). One location per request, so points are fetched by a small throttled thread pool with an overall
deadline. TTL 30 min, cached per point (rounded to 0.01 deg like Open-Meteo).
"""
from __future__ import annotations

import copy
import threading
import time
from concurrent.futures import ThreadPoolExecutor, wait
from datetime import datetime

import httpx

from .cache import TIMEOUT_S, USER_AGENT, UpstreamUnavailable, _stamp, combine_sources, default_cache, describe_error, http_get

URL = "https://api.met.no/weatherapi/locationforecast/2.0/complete"
PROVIDER = "api.met.no"
ATTRIBUTION = "Weather data from MET Norway (CC BY 4.0)"
TTL_S = 1800
KT_PER_MS = 1.943844
DEGRADED = ("gust_kt", "visibility_m", "cape_jkg", "wind250")
DEFAULT_CONTACT = "github.com/CypherMan2310/FlightRoute"
WORKERS = 12
MIN_INTERVAL_S = 0.06  # at most about 16 request starts per second across the pool
DEADLINE_S = 25.0  # stop waiting after this long: whatever is missing stays simulated
BACKOFF_S = 120  # after a batch in which every request failed, do not ask again for this long
GIVE_UP_AFTER = 5  # consecutive failures before any success: stop sending the rest of the batch


def user_agent(env=None) -> str:
    return f"{USER_AGENT} {(env or {}).get('AEROGRID_CONTACT') or DEFAULT_CONTACT}"


def _epoch(iso_z: str) -> int:
    return int(datetime.fromisoformat(iso_z.replace("Z", "+00:00")).timestamp())


def _normalise(body: dict, hours: int) -> dict:
    """Hourly rows in the Open-Meteo shape. Only steps that carry `next_1_hours` (hourly steps, about 60 h) are used."""
    rows = []
    for step in body["properties"]["timeseries"]:
        nxt = step["data"].get("next_1_hours")
        if nxt is None:
            continue
        d = step["data"]["instant"]["details"]
        wind = float(d["wind_speed"]) * KT_PER_MS
        rows.append({
            "time": _epoch(step["time"]), "temp_c": float(d["air_temperature"]), "precip_mm": float(nxt["details"]["precipitation_amount"]),
            "wind_kt": wind, "gust_kt": wind, "visibility_m": 10000.0, "cloud_low_pct": float(d["cloud_area_fraction_low"]), "cape_jkg": 0.0,
            "weather_code": None, "wind250_kt": 0.0, "wind250_dir_deg": 270.0,
        })
        if len(rows) == hours:
            break
    if not rows:
        raise ValueError("no hourly steps in the answer")
    return {"hourly": rows, "degraded": list(DEGRADED), "provider": PROVIDER}


class _Throttle:
    def __init__(self, min_interval: float):
        self.min_interval, self._next, self._lock = min_interval, 0.0, threading.Lock()

    def wait(self) -> None:
        with self._lock:
            now = time.monotonic()
            start = max(now, self._next)
            self._next = start + self.min_interval
        if start > now:
            time.sleep(start - now)


class _Breaker:
    """Stops a batch early when the service is clearly down (or has blocked us): saves 150 doomed requests and their latency."""

    def __init__(self, give_up_after: int):
        self.give_up_after, self.streak, self.ok, self._lock = give_up_after, 0, False, threading.Lock()

    @property
    def open(self) -> bool:
        return not self.ok and self.streak >= self.give_up_after

    def record(self, success: bool) -> None:
        with self._lock:
            if success:
                self.ok, self.streak = True, 0
            else:
                self.streak += 1


def get_points(points, hours: int = 24, *, client=None, cache=None, env=None, workers: int = WORKERS, deadline_s: float = DEADLINE_S,
               min_interval_s: float | None = None) -> tuple[dict[str, dict], str | None]:
    """({"lat,lon": record} for the points MET Norway could serve, first failure reason or None).

    A record is `live` (fetched now), `cached` (fresh) or `cached` + `stale` (the call failed, an older answer is reused for up to
    three TTLs). Points that cannot be served are simply absent: the caller keeps whatever it had (a simulation), it is never
    filled in here."""
    cache = cache if cache is not None else default_cache()
    keys = list(dict.fromkeys(f"{float(la):.2f},{float(lo):.2f}" for la, lo in points))
    now = cache.clock()
    out: dict[str, dict] = {}
    todo: list[str] = []
    for k in keys:
        hit = cache.get(f"metno:{hours}:{k}")
        if hit:
            out[k] = _stamp(copy.deepcopy(hit[0]), "cached", hit[0]["fetched_at"], now)
        else:
            todo.append(k)
    reasons: list[str] = []
    throttle, headers, breaker = _Throttle(MIN_INTERVAL_S if min_interval_s is None else min_interval_s), {"User-Agent": user_agent(env)}, _Breaker(GIVE_UP_AFTER)
    backoff = cache.get("metno:backoff") if todo else None

    own = client is None and bool(todo) and not backoff
    http = httpx.Client(timeout=httpx.Timeout(TIMEOUT_S), follow_redirects=True) if own else client  # one connection pool: no TLS handshake per point

    def one(k: str) -> dict:
        if breaker.open:
            raise UpstreamUnavailable("circuit_open")
        throttle.wait()
        lat, lon = k.split(",")
        try:
            rec = _normalise(http_get(URL, params={"lat": lat, "lon": lon}, headers=headers, client=http).json(), hours)
        except Exception:
            breaker.record(False)
            raise
        breaker.record(True)
        return rec

    if todo and backoff:
        reasons.append(backoff[0])  # asked and failed moments ago: serve stale records if any, simulate the rest, do not hammer the service
        for k in todo:
            stale = cache.get(f"metno:{hours}:{k}", stale=True)
            if stale:
                out[k] = _stamp(copy.deepcopy(stale[0]), "cached", stale[0]["fetched_at"], now, stale=True, fallback_reason=backoff[0])
    elif todo:
        pool = ThreadPoolExecutor(max_workers=max(1, workers))
        futures = {pool.submit(one, k): k for k in todo}
        done, _ = wait(futures, timeout=deadline_s)
        pool.shutdown(wait=False, cancel_futures=True)
        if own:
            http.close()
        for fut, k in futures.items():
            rec, reason = None, None
            if fut in done:
                try:
                    rec = fut.result()
                except Exception as exc:  # noqa: BLE001 - any upstream failure must degrade, not crash
                    reason = describe_error(exc)
            else:
                reason = "deadline"
            if rec is not None:
                t = cache.clock()
                rec = {**rec, "fetched_at": t}
                cache.set(f"metno:{hours}:{k}", copy.deepcopy(rec), TTL_S)
                out[k] = _stamp(rec, "live", t, t)
                continue
            reasons.append(reason or "unknown")
            stale = cache.get(f"metno:{hours}:{k}", stale=True)
            if stale:
                out[k] = _stamp(copy.deepcopy(stale[0]), "cached", stale[0]["fetched_at"], now, stale=True, fallback_reason=reason)
        if not breaker.ok and reasons:
            cache.set("metno:backoff", reasons[0], BACKOFF_S)
    return out, (reasons[0] if reasons else None)


def fill_simulated(payload: dict, points, hours: int, *, client=None, cache=None, env=None, **kw) -> dict:
    """Upgrade an Open-Meteo forecast payload: points left `simulated` are replaced by MET Norway records when it can serve them.

    The payload keeps Open-Meteo's `fallback_reason` (why the primary was not used), `source` becomes the least-real of its points,
    `provider` names every provider used, and `degraded` lists what the MET Norway records lack. With nothing to rescue, or
    nothing rescued, the payload is returned unchanged (and without a single MET Norway request in the first case)."""
    sim = [i for i, p in enumerate(payload["points"]) if p["source"] == "simulated"]
    if not sim:
        return payload
    got, _ = get_points([points[i] for i in sim], hours, client=client, cache=cache, env=env, **kw)
    if not got:
        return payload
    new = list(payload["points"])
    for i in sim:
        la, lo = float(points[i][0]), float(points[i][1])
        rec = got.get(f"{la:.2f},{lo:.2f}")
        if rec:
            new[i] = {**rec, "lat": la, "lon": lo}
    providers = list(dict.fromkeys((p.get("provider") or payload["provider"]) for p in new if p["source"] != "simulated"))
    degraded = sorted({f for p in new for f in p.get("degraded", [])})
    real = [p for p in new if p["source"] != "simulated"]
    return {**payload, "points": new, "source": combine_sources(p["source"] for p in new), "provider": " + ".join(providers) or payload["provider"],
            "fetched_at": min((p["fetched_at"] for p in real), default=payload["fetched_at"]), "age_s": max((p["age_s"] for p in new), default=0.0),
            "degraded": degraded, "attribution": ATTRIBUTION if degraded else None}
