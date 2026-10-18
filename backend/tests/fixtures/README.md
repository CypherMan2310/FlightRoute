# Test fixtures

Recorded on 2026-10-17 from the live free endpoints (small queries, user agent `AEROGRID-college-project/0.1`):

| file | endpoint | note |
|---|---|---|
| `metar_kjfk_kord_vidp.json` | aviationweather.gov `/api/data/metar?ids=KJFK,KORD,VIDP&format=json` | verbatim |
| `taf_kjfk.json` | aviationweather.gov `/api/data/taf?ids=KJFK&format=json` | verbatim |
| `nas_status.xml` | nasstatus.faa.gov `/api/airport-status-information` | verbatim; no ground stop / ground delay was active |
| `openmeteo_two_points.json` | api.open-meteo.com `/v1/forecast` (JFK, LHR, 3 hours, knots, 11 variables incl. low cloud, CAPE, 250 hPa wind) | verbatim |
| `openmeteo_hist_two_points.json` | historical-forecast-api.open-meteo.com `/v1/forecast` (2 points, 2026-05-01, knots) | verbatim |
| `adsblol_point.json` | api.adsb.lol `/v2/point/40.64/-73.78/50` | trimmed to 7 of 32 aircraft |
| `opensky_states.json` | opensky-network.org `/api/states/all` (anonymous, NYC box) | verbatim |
| `bts_sample.csv` | BTS On-Time Performance, May 2026 (the 17 columns we read) | 11 real rows picked to cover late / on-time / cancelled / diverted, Alaska, Hawaii, Phoenix |
| `iem_atl_sample.csv` | IEM ASOS `asos.py?station=ATL` (May 2026) | 9 real report lines: thunderstorm, vertical visibility, gust, clear |

CONSTRUCTED (hand-written from the documented schemas, not recorded): `nas_programs.constructed.xml` (ground delay / ground stop
element names are from the FAA feed convention, not from a live capture), `ebird_recent.constructed.json` (no eBird key was
available), `opensky_token.constructed.json`. Replace them with recordings when a live example is available.
