"""FAA NAS Status: US airport delay programmes (nasstatus.faa.gov/api/airport-status-information, XML). TTL 2 min.

No key and no published usage policy were found (checked 2026-10-17); the 2-minute TTL keeps us polite.
This is the ground truth for the live scorecard, US airports only. Closure entries are exposed as `notams`
and never counted as delay: the feed mixes GA-only restrictions and nightly curfews with real closures.
"""
from __future__ import annotations

import re
import xml.etree.ElementTree as ET
from datetime import datetime, timezone

from . import simulate
from .cache import default_cache, fetch_one, http_get

URL = "https://nasstatus.faa.gov/api/airport-status-information"
PROVIDER = "nasstatus.faa.gov"
TTL_S = 120
_MONTHS = {m: i for i, m in enumerate(("Jan", "Feb", "Mar", "Apr", "May", "Jun", "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"), 1)}
NORMAL = {"status": "normal", "delay_minutes": 0, "ground_stop": False, "ground_delay": False, "programs": []}


def parse_minutes(text: str | None) -> int | None:
    """'1 hour and 16 minutes' -> 76, '45 minutes' -> 45, '2 hours' -> 120; None if no duration is present."""
    if not text:
        return None
    h = re.search(r"(\d+)\s*hour", text, re.I)
    m = re.search(r"(\d+)\s*min", text, re.I)
    if not (h or m):
        return None
    return (int(h.group(1)) * 60 if h else 0) + (int(m.group(1)) if m else 0)


def _updated(text: str | None) -> int | None:
    m = re.search(r"(\w{3}) (\d{1,2}) (\d{2}):(\d{2}):(\d{2}) (\d{4})", text or "")
    if not m or m.group(1) not in _MONTHS:
        return None
    mon, day, hh, mm, ss, yr = m.groups()
    return int(datetime(int(yr), _MONTHS[mon], int(day), int(hh), int(mm), int(ss), tzinfo=timezone.utc).timestamp())


def summarise(programs: list[dict]) -> dict:
    """Airport record from its programmes. Precedence: ground_stop > ground_delay > delay."""
    kinds = {p["type"] for p in programs}
    stop, gdp = "ground_stop" in kinds, "ground_delay" in kinds
    return {
        "status": "ground_stop" if stop else "ground_delay" if gdp else "delay",
        "delay_minutes": max((p.get("max_minutes") or 0 for p in programs), default=0),
        "ground_stop": stop,
        "ground_delay": gdp,
        "programs": programs,
    }


def _text(el: ET.Element, tag: str) -> str | None:
    t = el.findtext(tag)
    return t.strip() if t and t.strip() else None


def parse_status(text: str) -> dict:
    if "<!ENTITY" in text:
        raise ValueError("entity declarations are not accepted")
    root = ET.fromstring(text)
    if root.tag != "AIRPORT_STATUS_INFORMATION":
        raise ValueError("not an AIRPORT_STATUS_INFORMATION document")
    by_airport: dict[str, list[dict]] = {}
    notams: list[dict] = []
    for dtype in root.iter("Delay_type"):  # the same section name can repeat; never key by it
        name = (dtype.findtext("Name") or "").lower()
        for container in dtype:
            if container.tag == "Name":
                continue
            for entry in container:
                airport = (_text(entry, "ARPT") or "").upper()
                if not airport:
                    continue
                reason = _text(entry, "Reason")
                if "closure" in name:
                    notams.append({"airport": airport, "reason": " ".join((reason or "").split()), "start": _text(entry, "Start"), "reopen": _text(entry, "Reopen")})
                elif "ground stop" in name:
                    prog = {"type": "ground_stop", "reason": reason, "end_time": _text(entry, "End_Time")}
                elif "ground delay" in name:
                    prog = {"type": "ground_delay", "reason": reason, "avg_minutes": parse_minutes(_text(entry, "Avg")), "max_minutes": parse_minutes(_text(entry, "Max"))}
                else:
                    ad = entry.find("Arrival_Departure")
                    src = ad if ad is not None else entry
                    prog = {
                        "type": "general_delay",
                        "direction": (ad.get("Type", "") if ad is not None else "").lower() or None,
                        "reason": reason,
                        "min_minutes": parse_minutes(_text(src, "Min")),
                        "max_minutes": parse_minutes(_text(src, "Max")),
                        "trend": _text(src, "Trend"),
                    }
                if "closure" not in name:
                    by_airport.setdefault(airport, []).append(prog)
    return {"updated": _updated(root.findtext("Update_Time")), "airports": {a: summarise(p) for a, p in by_airport.items()}, "notams": notams}


def get_status(*, client=None, cache=None) -> dict:
    cache = cache if cache is not None else default_cache()

    def fetch() -> dict:
        return {"provider": PROVIDER, **parse_status(http_get(URL, client=client).text)}

    return fetch_one(cache, "nas:status", ttl=TTL_S, fetch=fetch, simulate=lambda: simulate.sim_nas(cache.clock()))


def airport_status(payload: dict, code: str) -> dict:
    """Record for one airport; airports absent from the feed have no programme, i.e. status 'normal'."""
    return payload["airports"].get(code.strip().upper(), {**NORMAL, "programs": []})
