import io
from datetime import datetime, timezone
from pathlib import Path

import httpx
import numpy as np
import pandas as pd
import pytest

from aerogrid import data
from aerogrid.ml import congestion as cg
from aerogrid.ml import weather as wx
from aerogrid.train import real

from conftest import FIXTURES, fixture_json, fixture_text, mock_client


def utc(s):
    return pd.Timestamp(s, tz="UTC").timestamp()


def test_airport_tz_covers_every_us_seed_airport():
    us = {a.iata for a in data.load_airports() if a.country in {"US", "PR"}}
    assert set(real.AIRPORT_TZ) == us


def test_bts_candidate_urls_name_the_month():
    urls = real.candidate_urls(2026, 5)
    assert urls and all("2026_5" in u and u.startswith("https://") for u in urls)


def test_read_bts_converts_local_schedule_to_utc_and_applies_conventions():
    df = real.read_bts(FIXTURES / "bts_sample.csv", airports=set(real.AIRPORT_TZ))
    assert len(df) == 9, "the cancelled and the diverted flight are dropped"
    by = {(r.origin, r.dest, r.carrier, round(r.arr_delay)): r for r in df.itertuples()}
    jfk = by[("JFK", "LAX", "AA", 849)]
    assert jfk.ts_dep == utc("2026-05-08 11:59")  # 07:59 EDT
    assert jfk.ts_arr == jfk.ts_dep + 376 * 60 and jfk.crs_elapsed_min == 376
    assert jfk.late_aircraft_delay == 849 and jfk.carrier_delay == 0
    assert jfk.distance_km == pytest.approx(2475 * 1.609344)
    assert by[("PHX", "LAX", "OO", 39)].ts_dep == utc("2026-05-03 13:00")  # Phoenix has no DST: UTC-7
    assert by[("ANC", "DEN", "UA", 145)].ts_dep == utc("2026-06-01 07:09")  # 23:09 AKDT on May 31
    on_time = df[df["arr_delay"] == 0]
    assert len(on_time) == 3 and (on_time[["carrier_delay", "weather_delay", "nas_delay", "security_delay", "late_aircraft_delay"]] == 0).all().all()
    assert df.loc[df["origin"] == "LAX", "ts_dep"].max() == utc("2026-06-01 06:30")  # 23:30 PDT rolls into the next UTC day
    assert (df["ts_hour"] == np.floor(df["ts_dep"] / 3600) * 3600).all()
    assert df[["ts_dep", "ts_arr", "arr_delay", "dep_delay"]].notna().all().all()


def test_read_bts_filters_to_the_requested_airports():
    df = real.read_bts(FIXTURES / "bts_sample.csv", airports={"JFK", "LAX"})
    assert set(df["origin"]) | set(df["dest"]) <= {"JFK", "LAX"} and len(df) == 2


def test_read_bts_reads_a_zip_like_the_real_download(tmp_path):
    import zipfile

    z = tmp_path / "bts.zip"
    with zipfile.ZipFile(z, "w") as zf:
        zf.writestr("On_Time_Reporting_Carrier_On_Time_Performance_(1987_present)_2026_5.csv", fixture_text("bts_sample.csv"))
        zf.writestr("readme.html", "<html></html>")
    assert len(real.read_bts(z, airports=set(real.AIRPORT_TZ))) == 9


def test_parse_iem_builds_canonical_observations():
    obs = real.parse_iem(fixture_text("iem_atl_sample.csv"))
    assert list(obs.columns[:2]) == ["station", "ts"] and set(wx.OBS) <= set(obs.columns) and len(obs) == 9
    row = lambda stamp: obs[obs["ts"] == utc(stamp)].iloc[0]  # noqa: E731
    ts = row("2026-05-07 01:32")  # -TSRA, ceiling BKN at 4,500 ft
    assert (ts["obs_tstorm"], ts["obs_precip"], ts["obs_ceiling_kft"], ts["obs_vis_sm"]) == (1.0, 1.0, 4.5, 8.0)
    assert row("2026-05-17 20:20")["obs_precip"] == 2.0 and row("2026-05-17 20:20")["obs_cat"] == 2.0  # +TSRA, 1 SM: IFR
    vv = row("2026-05-10 11:52")  # vertical visibility 200 ft, fog, 0.5 SM
    assert (vv["obs_ceiling_kft"], vv["obs_cat"], vv["obs_vis_sm"]) == (0.2, 3.0, 0.5)
    assert row("2026-05-03 07:52")["obs_ceiling_kft"] == 12.0  # CLR
    g = row("2026-05-02 17:52")
    assert (g["obs_gust_kt"], g["obs_wind_kt"]) == (16.0, 9.0)
    assert row("2026-05-01 00:52")["obs_gust_kt"] == 7.0  # no gust reported: gust = wind


def test_parse_iem_handles_missing_values_and_empty_answers():
    text = "station,valid,vsby,sknt,gust,skyc1,skyc2,skyc3,skyc4,skyl1,skyl2,skyl3,skyl4,wxcodes,p01i\nJFK,2026-05-01 00:51,M,M,M,M,M,M,M,M,M,M,M,M,M\n"
    obs = real.parse_iem(text)
    assert len(obs) == 1 and obs.iloc[0]["obs_vis_sm"] == 10.0 and obs.iloc[0]["obs_wind_kt"] == 0.0
    assert len(real.parse_iem("station,valid,vsby,sknt,gust,skyc1,skyc2,skyc3,skyc4,skyl1,skyl2,skyl3,skyl4,wxcodes,p01i\n")) == 0


def test_hourly_obs_takes_the_latest_report_at_or_before_each_hour_and_neutral_when_stale():
    obs = real.parse_iem(fixture_text("iem_atl_sample.csv"))
    obs = obs[obs["ts"] < utc("2026-05-02")]  # three reports: 00:52, 01:52, 02:52
    hours = np.array([utc("2026-05-01 01:00"), utc("2026-05-01 02:00"), utc("2026-05-01 03:00"), utc("2026-05-01 06:00")])
    grid = real.hourly_obs(obs, hours)
    assert grid.shape == (4, len(wx.OBS)) and list(real.OBS_ORDER) == wx.OBS
    assert grid[0, 0] == 10.0  # 00:52 report used for 01:00
    assert grid[2, wx.OBS.index("obs_wind_kt")] == 8.0  # the 02:52 report (8 kt) is used for 03:00
    assert tuple(grid[3]) == tuple(wx.NEUTRAL_OBS[k] for k in wx.OBS)  # 06:00 is more than 90 minutes after the last report
    grid2, fresh = real.hourly_obs(obs, hours, return_mask=True)
    assert (grid2 == grid).all() and fresh.tolist() == [True, True, True, False]


def test_snap_cell_and_enroute_cells_use_a_two_degree_grid():
    assert real.snap_cell(40.64, -73.78) == (40.0, -74.0)
    assert real.snap_cell(41.1, -73.2) == (42.0, -74.0)
    jfk, lax = data.get_airport("JFK"), data.get_airport("LAX")
    cells = real.enroute_cells(jfk.lat, jfk.lon, lax.lat, lax.lon)
    assert len(cells) == 3 and all(c[0] % 2 == 0 and c[1] % 2 == 0 for c in cells)
    assert cells[0][1] > cells[1][1] > cells[2][1]  # westbound: longitude decreases along the path


def test_parse_forecast_gives_canonical_hourly_arrays():
    series = real.parse_forecast(fixture_json("openmeteo_hist_two_points.json"))
    assert len(series) == 2
    first = series[0]
    assert len(first["time"]) == 24 and first["time"][0] == utc("2026-05-01 00:00")
    assert set(real.FC_ORDER) == set(wx.FC) and first["fc"].shape == (24, len(wx.FC))
    assert first["wind250"].shape == (24, 2) and np.isfinite(first["wind250"]).all()
    raw = fixture_json("openmeteo_hist_two_points.json")[0]["hourly"]
    assert first["fc"][5, wx.FC.index("fc_gust_kt")] == raw["wind_gusts_10m"][5]
    assert first["fc"][5, wx.FC.index("fc_vis_km")] == min(raw["visibility"][5] / 1000, 10.0)


def test_split_for_window_keeps_the_24_3_3_month_proportions_with_seven_day_gaps():
    start, end = utc("2023-01-01"), utc("2023-01-01") + 30 * 30.4375 * 86400
    s = real.split_for_window(start, end)
    day = 86400
    assert s["calib_start"] - s["train_end"] == 7 * day and s["test_start"] - s["calib_end"] == 7 * day
    assert (s["train_end"] - start) / (end - start) == pytest.approx(0.8)
    assert (s["calib_end"] - start) / (end - start) == pytest.approx(0.9)


class Stub:
    """Offline stand-ins for the three downloads."""

    def __init__(self):
        self.calls = {"bts": [], "iem": [], "fc": []}

    def bts_available(self, year, month):
        return True

    def bts_frame(self, year, month, airports):
        self.calls["bts"].append((year, month))
        return real.read_bts(FIXTURES / "bts_sample.csv", airports=airports)

    def iem_text(self, station, start, end):
        self.calls["iem"].append(station)
        return fixture_text("iem_atl_sample.csv") if station == "ATL" else "station,valid,vsby,sknt,gust,skyc1,skyc2,skyc3,skyc4,skyl1,skyl2,skyl3,skyl4,wxcodes,p01i\n"

    def forecast_json(self, points, start, end):
        self.calls["fc"].append(len(points))
        times = pd.date_range(start, end, freq="h", tz="UTC", inclusive="left").strftime("%Y-%m-%dT%H:%M").tolist()
        n = len(times)
        return [{"hourly": {"time": times, "precipitation": [0.0] * n, "wind_gusts_10m": [12.0] * n, "visibility": [24000.0] * n, "cloud_cover_low": [10] * n,
                            "cape": [0.0] * n, "wind_speed_250hPa": [80.0] * n, "wind_direction_250hPa": [270] * n}} for _ in points]


def test_build_tables_from_stubbed_downloads_has_the_training_schema(tmp_path):
    stub = Stub()
    airports = ["ATL", "ORD", "LAX", "JFK", "PHX", "HNL", "ANC", "DEN"]
    t = real.load_real(months=1, end_month=(2026, 5), work_dir=tmp_path, sources=stub, airports=airports)
    assert t.meta["trained_on"] == "real" and "BTS" in t.meta["note"]
    assert 0.0 <= t.meta["obs_missing_frac"] <= 1.0 and t.meta["obs_missing_frac"] > 0.8, "only ATL has reports in the stub, the other 7 airports have none"
    assert stub.calls["bts"] == [(2026, 5)] and sorted(stub.calls["iem"]) == sorted(airports)
    f, ah = t.flights, t.airport_hour
    assert len(f) == 7, "9 flown flights, minus 2 whose UTC departure falls after the window end (late evening May 31 local)"
    assert len(ah) == len(airports) * 31 * 24
    assert not ah.duplicated(["airport", "ts"]).any() and ah.groupby("ts")["horizon_h"].nunique().eq(1).all()
    x = wx.build_flight_features(f)
    assert list(x.columns) == wx.FLIGHT_FEATURES and not x.isna().any().any()
    ah = ah.assign(sched_deps=1.0, sched_arrs=1.0, **{c: 0.0 for c in cg.HIST_FEATURES})
    assert list(cg.build_airport_features(ah).columns) == cg.AIRPORT_FEATURES
    assert {"route_extra_min", "bird_index", "o_zone", "ts_hour", "horizon_h"} <= set(f.columns)
    assert (f["route_extra_min"] >= 0).all() and f["bird_index"].between(0, 0.8).all()
    joined = ah.set_index(["airport", "ts"]).reindex(pd.MultiIndex.from_frame(f[["origin", "ts_hour"]]))
    assert not joined["horizon_h"].isna().any() and (joined["horizon_h"].to_numpy() == f["horizon_h"].to_numpy()).all()
    assert (ah.loc[ah["n"] == 0, ["y_dep", "y_late", "y_nas"]].isna()).all().all()
    s = t.split
    assert s["train_end"] < s["calib_start"] < s["calib_end"] < s["test_start"]


def test_load_real_rejects_a_window_before_open_meteo_history(tmp_path):
    with pytest.raises(ValueError, match="2022"):
        real.load_real(months=60, end_month=(2026, 5), work_dir=tmp_path, sources=Stub(), airports=["ATL", "ORD"])


def _http(tmp_path, handler):
    return real.HttpSources(tmp_path, client=mock_client(handler), sleep=lambda s: None)


def test_http_iem_text_builds_the_documented_query(tmp_path):
    seen = []

    def handler(request):
        seen.append(request)
        return httpx.Response(200, text=fixture_text("iem_atl_sample.csv"))

    text = _http(tmp_path, handler).iem_text("ATL", "2026-05-01", "2026-05-08")
    q = seen[0].url.params
    assert seen[0].url.host == "mesonet.agron.iastate.edu" and (q["station"], q["tz"], q["format"]) == ("ATL", "Etc/UTC", "onlycomma")
    assert q.get_list("report_type") == ["3", "4"] and {"vsby", "sknt", "gust", "skyc1", "skyl1", "wxcodes"} <= set(q.get_list("data"))
    assert text.startswith("station,valid")


def test_http_iem_splits_long_ranges_into_chunks_of_at_most_three_months_and_joins_them(tmp_path):
    seen = []

    def handler(request):
        seen.append(request.url.params["month1"] + "-" + request.url.params["month2"])
        return httpx.Response(200, text="station,valid,vsby\nATL,2026-05-01 00:52,10.00\n")

    text = _http(tmp_path, handler).iem_text("ATL", "2025-01-01", "2025-08-01")
    assert len(seen) == 3
    assert text.count("station,valid") == 1 and text.count("ATL,") == 3


def test_http_iem_retries_server_errors_then_gives_up(tmp_path):
    calls = []

    def handler(request):
        calls.append(1)
        return httpx.Response(503)

    with pytest.raises(httpx.HTTPStatusError):
        _http(tmp_path, handler).iem_text("ATL", "2026-05-01", "2026-05-08")
    assert len(calls) == 3


def _zip_bytes():
    import zipfile

    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w") as zf:
        zf.writestr("On_Time_Reporting_Carrier_On_Time_Performance_(1987_present)_2026_5.csv", fixture_text("bts_sample.csv"))
    return buf.getvalue()


def test_http_bts_tries_the_next_url_on_404_and_caches_the_zip_on_disk(tmp_path):
    seen = []

    def handler(request):
        seen.append(str(request.url))
        return httpx.Response(404) if request.url.host == "transtats.bts.gov" else httpx.Response(200, content=_zip_bytes())

    src = _http(tmp_path, handler)
    df = src.bts_frame(2026, 5, set(real.AIRPORT_TZ))
    assert len(df) == 9 and len(seen) == 2 and (tmp_path / "bts" / "bts_2026_05.zip").exists()
    src.bts_frame(2026, 5, set(real.AIRPORT_TZ))
    assert len(seen) == 2, "second call is served from the cached zip"
    assert not list((tmp_path / "bts").glob("*.part"))


def test_http_bts_availability_probe_uses_head_requests(tmp_path):
    methods = []

    def handler(request):
        methods.append(request.method)
        return httpx.Response(200 if "2026_5" in str(request.url) else 404)

    src = _http(tmp_path, handler)
    assert src.bts_available(2026, 5) is True and src.bts_available(2026, 9) is False
    assert set(methods) == {"HEAD"}


def test_http_forecast_caches_responses_on_disk_and_reports_rate_limits(tmp_path):
    calls = []
    body = fixture_json("openmeteo_hist_two_points.json")

    def ok(request):
        calls.append(request)
        return httpx.Response(200, json=body)

    pts = [(40.0, -74.0), (42.0, -88.0)]
    src = _http(tmp_path, ok)
    out = src.forecast_json(pts, "2026-05-01", "2026-05-02")
    assert len(out) == 2 and calls[0].url.params["latitude"] == "40.0,42.0" and calls[0].url.params["wind_speed_unit"] == "kn"
    assert calls[0].url.host == "historical-forecast-api.open-meteo.com" and "cape" in calls[0].url.params["hourly"]
    assert calls[0].url.params["end_date"] == "2026-05-01", "the end of the window is exclusive"
    src.forecast_json(pts, "2026-05-01", "2026-05-02")
    assert len(calls) == 1, "second call is served from the on-disk cache"
    limited = _http(tmp_path / "other", lambda r: httpx.Response(429))
    with pytest.raises(real.RateLimited):
        limited.forecast_json(pts, "2026-05-01", "2026-05-02")
