"""The artifacts committed in backend/models must load with the current code and be honestly labelled.

If this fails after a code change that alters a feature list or an artifact format, retrain (see TODO.md) and commit the new bundle.
"""
import json
from pathlib import Path

import pytest

from aerogrid.ml import ensemble as en

MODELS = Path(__file__).resolve().parents[1] / "models"


def card():
    return json.loads((MODELS / "model_card.json").read_text(encoding="utf-8"))


def test_the_committed_bundle_loads_with_the_current_code():
    c = card()
    if c.get("status") != "trained":
        pytest.skip("no trained bundle committed")
    assert en.Bundle.load_error(MODELS) is None


def test_the_committed_card_is_honest_about_what_it_was_trained_on():
    c = card()
    if c.get("status") != "trained":
        pytest.skip("no trained bundle committed")
    assert c["trained_on"] in {"real", "synthetic"}
    if c["trained_on"] == "synthetic":
        assert "SYNTHETIC" in c["note"].upper()
    else:
        assert "REAL" in c["note"].upper() and c["source_meta"]["months"]
    for name, m in c["models"].items():
        assert m["kind"] in {"ml", "algorithmic", "prior", "lookup", "stacked"}, name
        assert m["trained_on"] in {c["trained_on"], "n/a"}, name
    assert c["models"]["routing"]["label"] == "algorithmic (not ML)" and c["models"]["bird"]["label"] == "prior-based estimate"


def test_every_reported_number_comes_with_its_n_and_a_baseline():
    c = card()
    if c.get("status") != "trained":
        pytest.skip("no trained bundle committed")
    e = c["models"]["ensemble"]
    assert e["metrics"]["n"] > 0 and {"hist_mean", "lgbm_direct"} <= set(e["baselines"])
    assert all(v == "insufficient data" or v["n"] >= 1000 for v in e["per_zone"].values()), "thin zones must say so instead of reporting a number"
