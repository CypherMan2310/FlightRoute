"""Phase 0 checks: package imports and seed airports CSV is well-formed."""
import csv
import importlib
import re
from pathlib import Path

SEED = Path(__file__).resolve().parents[1] / "data" / "airports_seed.csv"
COLUMNS = ["iata", "icao", "name", "city", "country", "lat", "lon", "tier", "notes"]
MODULES = [
    "aerogrid.data",
    *(f"aerogrid.sources.{m}" for m in ["adsb", "metar", "openmeteo", "nas", "ebird", "cache", "simulate"]),
    *(f"aerogrid.ml.{m}" for m in ["gcn", "congestion", "weather", "bird", "routing", "ensemble", "conformal", "explain", "whatif"]),
    *(f"aerogrid.train.{m}" for m in ["synth", "real", "train_all"]),
    *(f"aerogrid.api.{m}" for m in ["main", "scorecard", "schemas"]),
]


def _rows():
    with SEED.open(newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        assert reader.fieldnames == COLUMNS
        return list(reader)


def test_modules_import():
    for name in MODULES:
        importlib.import_module(name)


def test_seed_shape_and_codes():
    rows = _rows()
    assert 140 <= len(rows) <= 170
    iata = [r["iata"] for r in rows]
    icao = [r["icao"] for r in rows]
    assert len(set(iata)) == len(iata)
    assert len(set(icao)) == len(icao)
    assert all(re.fullmatch(r"[A-Z]{3}", c) for c in iata)
    assert all(re.fullmatch(r"[A-Z]{4}", c) for c in icao)
    assert all(re.fullmatch(r"[A-Z]{2}", r["country"]) for r in rows)


def test_seed_values():
    for r in _rows():
        assert -90 <= float(r["lat"]) <= 90, r["iata"]
        assert -180 <= float(r["lon"]) <= 180, r["iata"]
        assert r["tier"] in {"1", "2", "3"}, r["iata"]
        assert r["notes"] in {"", "?"}, r["iata"]
        if r["country"] == "US":
            assert r["icao"][0] in "KP", r["iata"]
