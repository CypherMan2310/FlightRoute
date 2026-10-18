"""REAL training tables: BTS On-Time Performance + IEM ASOS observations + Open-Meteo historical forecasts -> Tables.

Honest scope: US airports of the seed list only (both ends of a flight). Downloads are slow (a BTS month is ~30 MB at ~350 kB/s,
the Open-Meteo calls are budgeted against the free 10,000/day limit) so everything is cached under `work_dir` and the run is
resumable; real runs belong in GitHub Actions, not in a sandbox.

Rules from docs/phase2-design.md:
  * BTS times are local: scheduled departure UTC = FlightDate + CRSDepTime in the origin's time zone; DepTime / ArrTime are never parsed.
  * Features are those that exist live at issue time t - h (METAR-style obs at t - h, forecast for the target hour); cause columns,
    NAS status and anything computed from flights at or after t are never features.
  * Cancelled / diverted flights are excluded; cause columns are 0-filled when ArrDel15 = 0 (BTS leaves them blank).
"""
from __future__ import annotations

import hashlib
import io
import json
import sys
import time
import zipfile
from dataclasses import dataclass
from datetime import date, datetime, timedelta, timezone
from pathlib import Path
from typing import Callable

import httpx
import numpy as np
import pandas as pd

from .. import data
from ..ml import bird, routing
from ..ml import weather as wx
from ..sources import cache as net
from ..sources.metar import flight_category
from . import Tables

KM_PER_MILE = 1.609344
MIN_START = datetime(2022, 6, 1, tzinfo=timezone.utc)  # earliest date verified in the Open-Meteo historical-forecast API
HOUR, DAY = 3600, 86400
GRID_DEG = 2.0
BTS_COLS = ["FlightDate", "Reporting_Airline", "Origin", "Dest", "CRSDepTime", "CRSElapsedTime", "DepDelayMinutes", "ArrDelayMinutes", "ArrDel15",
            "Cancelled", "Diverted", "Distance", "CarrierDelay", "WeatherDelay", "NASDelay", "SecurityDelay", "LateAircraftDelay"]
CAUSES = {"CarrierDelay": "carrier_delay", "WeatherDelay": "weather_delay", "NASDelay": "nas_delay", "SecurityDelay": "security_delay", "LateAircraftDelay": "late_aircraft_delay"}
IEM_DATA = ["vsby", "sknt", "gust", "skyc1", "skyc2", "skyc3", "skyc4", "skyl1", "skyl2", "skyl3", "skyl4", "wxcodes", "p01i"]
OM_HOURLY = ["precipitation", "wind_gusts_10m", "visibility", "cloud_cover_low", "cape", "wind_speed_250hPa", "wind_direction_250hPa"]
OBS_ORDER, FC_ORDER = tuple(wx.OBS), tuple(wx.FC)

AIRPORT_TZ = {
    "ATL": "America/New_York", "BOS": "America/New_York", "BWI": "America/New_York", "CLE": "America/New_York", "CLT": "America/New_York",
    "DCA": "America/New_York", "DTW": "America/New_York", "EWR": "America/New_York", "FLL": "America/New_York", "IAD": "America/New_York",
    "JFK": "America/New_York", "LGA": "America/New_York", "MCO": "America/New_York", "MIA": "America/New_York", "PHL": "America/New_York",
    "RDU": "America/New_York", "TPA": "America/New_York", "DFW": "America/Chicago", "AUS": "America/Chicago", "BNA": "America/Chicago",
    "DAL": "America/Chicago", "HOU": "America/Chicago", "IAH": "America/Chicago", "MCI": "America/Chicago", "MDW": "America/Chicago",
    "MSP": "America/Chicago", "MSY": "America/Chicago", "ORD": "America/Chicago", "SAT": "America/Chicago", "STL": "America/Chicago",
    "DEN": "America/Denver", "SLC": "America/Denver", "PHX": "America/Phoenix", "LAX": "America/Los_Angeles", "LAS": "America/Los_Angeles",
    "PDX": "America/Los_Angeles", "SAN": "America/Los_Angeles", "SEA": "America/Los_Angeles", "SFO": "America/Los_Angeles",
    "ANC": "America/Anchorage", "HNL": "Pacific/Honolulu", "SJU": "America/Puerto_Rico",
}


class RateLimited(Exception):
    """Open-Meteo answered 429: progress is cached on disk, wait for the limit to reset and re-run."""


# ---- BTS --------------------------------------------------------------------------------------------------------

def candidate_urls(year: int, month: int) -> list[str]:
    name = f"On_Time_Reporting_Carrier_On_Time_Performance_1987_present_{year}_{month}.zip"
    return [f"https://transtats.bts.gov/PREZIP/{name}", f"https://www.transtats.bts.gov/PREZIP/{name}"]


def _local_to_utc(dates: pd.Series, hhmm: pd.Series, airports: pd.Series) -> np.ndarray:
    hhmm = hhmm.astype(int)
    local = pd.to_datetime(dates) + pd.to_timedelta((hhmm // 100) % 24, unit="h") + pd.to_timedelta(hhmm % 100, unit="m")
    out = np.full(len(local), np.nan)
    tz = airports.map(AIRPORT_TZ)
    for name in tz.dropna().unique():
        m = (tz == name).to_numpy()
        loc = pd.DatetimeIndex(local[m]).tz_localize(name, ambiguous="NaT", nonexistent="shift_forward")
        out[m] = ((loc.tz_convert("UTC") - pd.Timestamp("1970-01-01", tz="UTC")) / pd.Timedelta(seconds=1)).to_numpy()
    return out


def read_bts(path, airports) -> pd.DataFrame:
    """One BTS month (zip or csv) -> flown flights between `airports`, with UTC scheduled times and BTS cause conventions applied."""
    path = Path(path)
    if zipfile.is_zipfile(path):
        with zipfile.ZipFile(path) as z:
            name = next(n for n in z.namelist() if n.lower().endswith(".csv"))
            with z.open(name) as f:
                df = pd.read_csv(f, usecols=BTS_COLS, low_memory=False, encoding="utf-8-sig")
    else:
        df = pd.read_csv(path, usecols=BTS_COLS, low_memory=False, encoding="utf-8-sig")
    airports = set(airports)
    df = df[df["Origin"].isin(airports) & df["Dest"].isin(airports) & (df["Cancelled"] == 0) & (df["Diverted"] == 0)]
    df = df.dropna(subset=["ArrDelayMinutes", "DepDelayMinutes", "CRSElapsedTime", "CRSDepTime", "FlightDate"]).reset_index(drop=True)
    ts_dep = _local_to_utc(df["FlightDate"], df["CRSDepTime"], df["Origin"])
    out = pd.DataFrame({
        "origin": df["Origin"].to_numpy(), "dest": df["Dest"].to_numpy(), "carrier": df["Reporting_Airline"].to_numpy(),
        "ts_dep": ts_dep, "crs_elapsed_min": df["CRSElapsedTime"].to_numpy(dtype=float), "distance_km": df["Distance"].to_numpy(dtype=float) * KM_PER_MILE,
        "arr_delay": df["ArrDelayMinutes"].to_numpy(dtype=float), "dep_delay": df["DepDelayMinutes"].to_numpy(dtype=float),
        **{new: df[old].fillna(0.0).to_numpy(dtype=float) for old, new in CAUSES.items()},
    })
    out = out[out["ts_dep"].notna()].reset_index(drop=True)
    out["ts_arr"] = out["ts_dep"] + out["crs_elapsed_min"] * 60
    out["ts_hour"] = np.floor(out["ts_dep"] / HOUR) * HOUR
    return out


# ---- IEM ASOS observations ------------------------------------------------------------------------------------

def parse_iem(text: str) -> pd.DataFrame:
    """IEM ASOS csv -> station, ts (epoch s) and the canonical obs_* columns (same function the live METAR path uses)."""
    cols = ["station", "ts", *wx.OBS]
    df = pd.read_csv(io.StringIO(text), na_values=["M"], keep_default_na=False, dtype=str)
    if df.empty:
        return pd.DataFrame(columns=cols)
    ts = (pd.to_datetime(df["valid"], utc=True) - pd.Timestamp("1970-01-01", tz="UTC")) / pd.Timedelta(seconds=1)
    num = lambda c: pd.to_numeric(df[c], errors="coerce")  # noqa: E731
    vsby, sknt, gust = num("vsby"), num("sknt"), num("gust")
    layers = [(df[f"skyc{i}"].fillna("").str.strip(), num(f"skyl{i}")) for i in range(1, 5)]
    rows = []
    for k in range(len(df)):
        bases = [lvl[k] for cover, lvl in layers if cover.iloc[k] in {"BKN", "OVC", "VV"} and lvl.iloc[k] == lvl.iloc[k] for lvl in [lvl.to_numpy()]]
        ceiling = min(bases) if bases else None
        vis = None if vsby.iloc[k] != vsby.iloc[k] else float(vsby.iloc[k])
        wind = 0.0 if sknt.iloc[k] != sknt.iloc[k] else float(sknt.iloc[k])
        g = None if gust.iloc[k] != gust.iloc[k] else float(gust.iloc[k])
        wxs = df["wxcodes"].iloc[k]
        rec = {"visibility_sm": vis, "ceiling_ft": ceiling, "wind_kt": wind, "gust_kt": g, "wx": wxs if isinstance(wxs, str) else "", "flight_category": flight_category(vis, ceiling)}
        rows.append(wx.obs_features(rec))
    out = pd.DataFrame(rows, columns=wx.OBS)
    out.insert(0, "ts", ts.to_numpy())
    out.insert(0, "station", df["station"].to_numpy())
    return out.drop_duplicates(["station", "ts"]).reset_index(drop=True)


def hourly_obs(obs: pd.DataFrame, hours, max_age_s: float = 5400.0, return_mask: bool = False):
    """(len(hours), n_obs): the latest report at or before each hour, neutral good weather if none is newer than `max_age_s`.
    With return_mask also returns a boolean array: True where a fresh report was found."""
    hours = np.asarray(hours, dtype=float)
    out = np.tile(np.array([wx.NEUTRAL_OBS[k] for k in OBS_ORDER]), (len(hours), 1))
    ok = np.zeros(len(hours), dtype=bool)
    if len(obs):
        o = obs.sort_values("ts")
        ts, vals = o["ts"].to_numpy(dtype=float), o[list(OBS_ORDER)].to_numpy(dtype=float)
        idx = np.searchsorted(ts, hours, side="right") - 1
        ok = (idx >= 0) & (hours - ts[np.clip(idx, 0, None)] <= max_age_s)
        out[ok] = vals[idx[ok]]
    return (out, ok) if return_mask else out


# ---- Open-Meteo historical forecast ----------------------------------------------------------------------------

def snap_cell(lat: float, lon: float, deg: float = GRID_DEG) -> tuple[float, float]:
    return float(np.round(lat / deg) * deg), float(np.round(lon / deg) * deg)


def enroute_cells(o_lat, o_lon, d_lat, d_lon) -> list[tuple[float, float]]:
    return [snap_cell(*data.great_circle_point(o_lat, o_lon, d_lat, d_lon, f)) for f in (0.25, 0.5, 0.75)]


def parse_forecast(raw) -> list[dict]:
    """Open-Meteo answer (list or single object) -> per point: time (epoch s), fc (T x 5, canonical order), wind250 (T x 2: kt, deg from)."""
    out = []
    for r in raw if isinstance(raw, list) else [raw]:
        h = r["hourly"]
        arr = lambda k: np.array([np.nan if v is None else v for v in h[k]], dtype=float)  # noqa: E731
        t = (pd.to_datetime(h["time"], utc=True) - pd.Timestamp("1970-01-01", tz="UTC")) / pd.Timedelta(seconds=1)
        fc = np.column_stack([arr("precipitation"), arr("wind_gusts_10m"), arr("cloud_cover_low"), np.minimum(arr("visibility") / 1000.0, 10.0), arr("cape")])
        out.append({"time": t.to_numpy(), "fc": fc, "wind250": np.column_stack([arr("wind_speed_250hPa"), arr("wind_direction_250hPa")])})
    return out


def split_for_window(start: float, end: float) -> dict:
    """train 80% / calib 10% / test 10% of the window (24 / 3 / 3 months of 30), separated by gaps of 7 days (less for short windows)."""
    span = end - start
    gap = min(7 * DAY, 0.02 * span)
    return {"train_end": start + 0.8 * span, "calib_start": start + 0.8 * span + gap, "calib_end": start + 0.9 * span, "test_start": start + 0.9 * span + gap}


# ---- downloads ---------------------------------------------------------------------------------------------------

class HttpSources:
    """The three real downloads. Every call has the shared 8 s timeout; failures retry (not 404 / 429); results are cached in work_dir."""

    def __init__(self, work_dir, client: httpx.Client | None = None, retries: int = 3, sleep: Callable[[float], None] = time.sleep):
        self.work = Path(work_dir)
        self.client = client or httpx.Client(timeout=httpx.Timeout(net.TIMEOUT_S), follow_redirects=True)
        self.retries, self.sleep = retries, sleep

    def _retry(self, fn):
        for attempt in range(self.retries):
            try:
                return fn()
            except (httpx.TransportError, httpx.HTTPStatusError) as exc:
                transient = isinstance(exc, httpx.TransportError) or exc.response.status_code >= 500
                if not transient or attempt == self.retries - 1:
                    raise
                self.sleep(2**attempt)

    def bts_available(self, year: int, month: int) -> bool:
        for url in candidate_urls(year, month):
            try:
                net._request("HEAD", url, client=self.client)
                return True
            except (httpx.HTTPStatusError, httpx.TransportError):
                continue
        return False

    def bts_frame(self, year: int, month: int, airports) -> pd.DataFrame:
        zip_path = self.work / "bts" / f"bts_{year}_{month:02d}.zip"
        if not zip_path.exists():
            zip_path.parent.mkdir(parents=True, exist_ok=True)
            part, last = zip_path.with_suffix(".part"), None
            for url in candidate_urls(year, month):
                try:
                    def download():
                        with self.client.stream("GET", url, headers={"User-Agent": net.USER_AGENT}) as r:
                            r.raise_for_status()
                            with part.open("wb") as f:
                                for chunk in r.iter_bytes(1 << 20):
                                    f.write(chunk)

                    self._retry(download)
                    part.replace(zip_path)
                    break
                except (httpx.HTTPStatusError, httpx.TransportError) as exc:
                    last = exc
                    part.unlink(missing_ok=True)
            else:
                raise last
        return read_bts(zip_path, airports)

    def iem_text(self, station: str, start: str, end: str) -> str:
        """ASOS reports for [start, end) (YYYY-MM-DD), in chunks of at most 92 days."""
        pieces, s, stop = [], date.fromisoformat(start), date.fromisoformat(end)
        while s < stop:
            e = min(s + timedelta(days=92), stop)
            e1 = e + timedelta(days=1)  # the service's end date is not reliably exclusive: ask one day more, duplicates are dropped when parsing
            params = {"station": station, "data": IEM_DATA, "year1": s.year, "month1": s.month, "day1": s.day, "year2": e1.year, "month2": e1.month, "day2": e1.day,
                      "tz": "Etc/UTC", "format": "onlycomma", "latlon": "no", "missing": "M", "trace": "T", "direct": "no", "report_type": [3, 4]}
            text = self._retry(lambda: net.http_get("https://mesonet.agron.iastate.edu/cgi-bin/request/asos.py", params=params, client=self.client).text)
            lines = text.splitlines()
            pieces.extend(lines if not pieces else lines[1:])
            s = e
        return "\n".join(pieces) + "\n"

    def forecast_json(self, points, start: str, end: str) -> list[dict]:
        """Open-Meteo historical forecast for each (lat, lon), window [start, end), cached per request on disk."""
        cache_dir = self.work / "openmeteo"
        cache_dir.mkdir(parents=True, exist_ok=True)
        per_point: list[dict | None] = [None] * len(points)
        s0, stop = date.fromisoformat(start), date.fromisoformat(end)
        for i in range(0, len(points), 8):
            chunk = points[i : i + 8]
            merged: list[dict] | None = None
            s = s0
            while s < stop:
                e = min(s + timedelta(days=92), stop)
                key = hashlib.sha1(f"{chunk}|{s}|{e}".encode()).hexdigest()[:20]
                f = cache_dir / f"{key}.json"
                if f.exists():
                    rows = json.loads(f.read_text(encoding="utf-8"))
                else:
                    params = {"latitude": ",".join(str(p[0]) for p in chunk), "longitude": ",".join(str(p[1]) for p in chunk), "hourly": ",".join(OM_HOURLY),
                              "start_date": s.isoformat(), "end_date": (e - timedelta(days=1)).isoformat(), "wind_speed_unit": "kn", "timezone": "UTC"}
                    try:
                        body = self._retry(lambda: net.http_get("https://historical-forecast-api.open-meteo.com/v1/forecast", params=params, client=self.client).json())
                    except httpx.HTTPStatusError as exc:
                        if exc.response.status_code == 429:
                            raise RateLimited("Open-Meteo free-tier limit reached; downloaded chunks are cached in the work dir, re-run later") from exc
                        raise
                    rows = body if isinstance(body, list) else [body]
                    f.write_text(json.dumps(rows), encoding="utf-8")
                if merged is None:
                    merged = rows
                else:
                    for m, r in zip(merged, rows):
                        for k, v in r["hourly"].items():
                            m["hourly"][k] = m["hourly"][k] + v
                s = e
            for j, r in enumerate(merged or []):
                per_point[i + j] = r
        return per_point


# ---- assembly ----------------------------------------------------------------------------------------------------

def _month_list(end_month: tuple[int, int], months: int) -> list[tuple[int, int]]:
    y, m = end_month
    out = []
    for _ in range(months):
        out.append((y, m))
        y, m = (y - 1, 12) if m == 1 else (y, m - 1)
    return out[::-1]


def _latest_available(sources, today: date) -> tuple[int, int]:
    y, m = today.year, today.month
    for _ in range(8):  # BTS publishes about 2-3 months behind
        y, m = (y - 1, 12) if m == 1 else (y, m - 1)
        if sources.bts_available(y, m):
            return y, m
    raise RuntimeError("no recent BTS month is available")


def _fill(a: np.ndarray, neutral: dict, order) -> np.ndarray:
    """Replace NaN with the neutral value of each column."""
    out = a.copy()
    for j, k in enumerate(order):
        out[..., j] = np.where(np.isnan(out[..., j]), neutral[k], out[..., j])
    return out


def load_real(months: int = 30, end_month: tuple[int, int] | None = None, work_dir=".aerogrid-work", sources=None, airports=None, seed: int = 0,
              log: Callable[..., None] = lambda *a: None) -> Tables:
    work_dir = Path(work_dir)
    sources = sources or HttpSources(work_dir)
    airports = sorted(airports or AIRPORT_TZ)
    end_month = end_month or _latest_available(sources, date.today())
    month_list = _month_list(end_month, months)
    t0 = datetime(*month_list[0], 1, tzinfo=timezone.utc).timestamp()
    ny, nm = (end_month[0] + 1, 1) if end_month[1] == 12 else (end_month[0], end_month[1] + 1)
    t_end = datetime(ny, nm, 1, tzinfo=timezone.utc).timestamp()
    if t0 < MIN_START.timestamp():
        raise ValueError(f"window starts {_d(t0)}, before 2022-06-01, the earliest date verified in the Open-Meteo historical-forecast API")
    hours = t0 + HOUR * np.arange(int((t_end - t0) // HOUR))
    n_hours, n_ap = len(hours), len(airports)
    aidx = {a: i for i, a in enumerate(airports)}
    start_s, end_s = _d(t0), _d(t_end)

    frames = []
    for y, m in month_list:
        log(f"BTS {y}-{m:02d}")
        frames.append(sources.bts_frame(y, m, set(airports)))
    fl = pd.concat(frames, ignore_index=True)
    fl = fl[(fl["ts_hour"] >= t0) & (fl["ts_hour"] < t_end)].reset_index(drop=True)
    fl["flight_id"] = np.arange(len(fl))

    obs_arr = np.zeros((n_ap, n_hours, len(OBS_ORDER)))
    obs_fresh = np.zeros((n_ap, n_hours), dtype=bool)
    for a in airports:
        log(f"IEM {a}")
        obs_arr[aidx[a]], obs_fresh[aidx[a]] = hourly_obs(parse_iem(sources.iem_text(a, start_s, end_s)), hours, return_mask=True)

    lat = {a: data.get_airport(a).lat for a in airports}
    lon = {a: data.get_airport(a).lon for a in airports}
    pairs = fl[["origin", "dest"]].drop_duplicates()
    pair_cells = {(o, d): [snap_cell(*c) for c in enroute_cells(lat[o], lon[o], lat[d], lon[d])] for o, d in zip(pairs["origin"], pairs["dest"])}
    cells = sorted({c for cs in pair_cells.values() for c in cs})
    cidx = {c: i for i, c in enumerate(cells)}
    points = [(lat[a], lon[a]) for a in airports] + cells
    log(f"Open-Meteo {len(points)} points")
    series = parse_forecast(sources.forecast_json(points, start_s, end_s))
    fc_all = np.full((len(points), n_hours, len(FC_ORDER)), np.nan, dtype=np.float32)
    w250_all = np.full((len(points), n_hours, 2), np.nan, dtype=np.float32)
    for p, s in enumerate(series):
        k = ((s["time"] - t0) // HOUR).astype(int)
        ok = (k >= 0) & (k < n_hours)
        fc_all[p, k[ok]], w250_all[p, k[ok]] = s["fc"][ok], s["wind250"][ok]
    fc_ap, fc_cell, w250_cell = fc_all[:n_ap], fc_all[n_ap:], w250_all[n_ap:]
    obs_missing = float(1.0 - obs_fresh.mean())
    fc_missing = float(np.isnan(fc_all).mean())
    fc_ap, fc_cell = _fill(fc_ap, wx.NEUTRAL_FC, FC_ORDER), _fill(fc_cell, wx.NEUTRAL_FC, FC_ORDER)

    rng = np.random.default_rng(seed)
    horizon = rng.choice([2, 6, 24], n_hours)  # one issue-time horizon per snapshot
    issue = np.clip(np.arange(n_hours) - horizon, 0, None)

    o, d = fl["origin"].map(aidx).to_numpy(), fl["dest"].map(aidx).to_numpy()
    t_idx = ((fl["ts_hour"].to_numpy() - t0) // HOUR).astype(int)
    arr_idx = np.clip(((fl["ts_arr"].to_numpy() - t0) // HOUR).astype(int), 0, n_hours - 1)
    iss = issue[t_idx]
    cols = {c: fl[c].to_numpy() for c in fl.columns}
    cols["horizon_h"] = horizon[t_idx]
    f_lat = np.array([lat[a] for a in airports])
    f_lon = np.array([lon[a] for a in airports])
    tier = np.array([data.get_airport(a).tier for a in airports])
    cols["o_lon"], cols["o_tier"], cols["d_tier"] = f_lon[o], tier[o], tier[d]
    months_arr = pd.DatetimeIndex(pd.to_datetime(fl["ts_dep"], unit="s")).month.to_numpy()
    zone_of = {(i, mth): data.weather_zone(float(f_lat[i]), float(f_lon[i]), int(mth)) for i in range(n_ap) for mth in range(1, 13)}
    cols["o_zone"] = np.array([zone_of[(i, mth)] for i, mth in zip(o, months_arr)])
    for j, k in enumerate(OBS_ORDER):
        cols[f"o_{k}"], cols[f"d_{k}"] = obs_arr[o, iss, j], obs_arr[d, iss, j]
    for j, k in enumerate(FC_ORDER):
        cols[f"o_{k}"], cols[f"d_{k}"] = fc_ap[o, t_idx, j], fc_ap[d, arr_idx, j]

    pair_ids = {p: n for n, p in enumerate(pair_cells)}
    pair_arr = np.array([pair_ids[(a, b)] for a, b in zip(fl["origin"], fl["dest"])])
    pair_cell_idx = np.array([[cidx[c] for c in cs] for cs in pair_cells.values()])  # (n_pairs, 3)
    er_fc, er_speed, er_from = [], [], []
    for k, frac in enumerate((0.25, 0.5, 0.75)):
        c = pair_cell_idx[pair_arr, k]
        h = np.clip(((fl["ts_dep"].to_numpy() + fl["crs_elapsed_min"].to_numpy() * 60 * frac - t0) // HOUR).astype(int), 0, n_hours - 1)
        er_fc.append(fc_cell[c, h])
        w = w250_cell[c, h]
        er_speed.append(np.nan_to_num(w[:, 0], nan=0.0))
        er_from.append(np.nan_to_num(w[:, 1], nan=270.0))
    er = np.stack(er_fc)  # (3, n, 5)
    cols["er_max_gust_kt"], cols["er_max_precip_mm"] = er[..., FC_ORDER.index("fc_gust_kt")].max(axis=0), er[..., FC_ORDER.index("fc_precip_mm")].max(axis=0)
    cols["er_max_cape_jkg"], cols["er_max_cloud_low_pct"] = er[..., FC_ORDER.index("fc_cape_jkg")].max(axis=0), er[..., FC_ORDER.index("fc_cloud_low_pct")].max(axis=0)
    cols["route_extra_min"] = routing.wind_extra_minutes_vec(f_lat[o], f_lon[o], f_lat[d], f_lon[d], np.stack(er_speed), np.stack(er_from))["extra_min"]
    cols["bird_index"] = bird.flight_bird_index(f_lat[o], f_lon[o], cols["ts_dep"], f_lat[d], f_lon[d], cols["ts_arr"])
    flights = pd.DataFrame(cols)

    grid_a, grid_t = np.meshgrid(np.arange(n_ap), np.arange(n_hours), indexing="ij")
    ga, gt = grid_a.ravel(), grid_t.ravel()
    ah = pd.DataFrame({"airport": np.array(airports)[ga], "ts": hours[gt], "horizon_h": horizon[gt], "tier": tier[ga], "lon": f_lon[ga]})
    for j, k in enumerate(OBS_ORDER):
        ah[k] = obs_arr[ga, issue[gt], j]
    for j, k in enumerate(FC_ORDER):
        ah[k] = fc_ap[ga, gt, j]
    amonth = pd.DatetimeIndex(pd.to_datetime(ah["ts"], unit="s")).month.to_numpy()
    ah["zone"] = [zone_of[(i, mth)] for i, mth in zip(ga, amonth)]
    agg = flights.groupby(["origin", "ts_hour"]).agg(n=("flight_id", "size"), y_dep=("dep_delay", "mean"), y_late=("late_aircraft_delay", "mean"), y_nas=("nas_delay", "mean"))
    agg.index.names = ["airport", "ts"]
    ah = ah.join(agg, on=["airport", "ts"])
    ah["n"] = ah["n"].fillna(0).astype(int)

    meta = {
        "trained_on": "real", "months": [f"{y}-{m:02d}" for y, m in month_list], "window": [start_s, end_s],
        "note": "REAL: BTS On-Time Performance (US domestic, seed airports only) + IEM ASOS observations + Open-Meteo historical forecast.",
        "n_flights": int(len(flights)), "obs_missing_frac": obs_missing, "fc_missing_frac": fc_missing,
        "limitations": ["US airports of the seed list only", "cancelled and diverted flights excluded", "BTS NAS delay includes non-extreme weather",
                        "cause minutes exist only when ArrDel15 = 1", "local time approximated by solar time in features"],
    }
    airports_df = pd.DataFrame({"iata": airports, "tier": tier, "lat": f_lat, "lon": f_lon})
    return Tables(flights, ah, airports_df, list(airports), split_for_window(t0, t_end), meta)


def _d(ts: float) -> str:
    return datetime.fromtimestamp(ts, tz=timezone.utc).strftime("%Y-%m-%d")
