import pytest

from aerogrid.api.scorecard import EVENT_MIN, Scorecard, wilson

T0 = 1_792_000_000.0  # issue time


def nas(source="live", age=10.0, **airports):
    return {"source": source, "age_s": age, "airports": {k: {"status": v[0], "delay_minutes": v[1]} for k, v in airports.items()}}


def row(airport, pred, event_at_issue=False, target=T0 + 7200, issued=T0):
    return {"airport": airport, "issued_at": issued, "target_at": target, "horizon_h": 2, "pred_dep_min": pred, "nas_event_at_issue": event_at_issue, "trained_on": "synthetic"}


@pytest.fixture
def sc():
    return Scorecard(":memory:")


def test_event_threshold_is_fifteen_minutes():
    assert EVENT_MIN == 15


def test_log_is_idempotent_per_airport_and_target_hour(sc):
    assert sc.log([row("JFK", 20), row("ATL", 5)]) == 2
    assert sc.log([row("JFK", 99)]) == 0, "same airport and target hour is not logged twice"
    assert sc.metrics(now=T0)["n_pending"] == 2


def test_resolution_uses_live_nas_data_only(sc):
    sc.log([row("JFK", 20)])
    now = T0 + 7200 + 60
    assert sc.resolve(nas("simulated", JFK=("delay", 45)), now) == 0, "simulated NAS data must never resolve a prediction"
    assert sc.resolve(nas("cached", age=900, JFK=("delay", 45)), now) == 0, "a stale cached answer must not resolve either"
    assert sc.resolve(nas("live", JFK=("delay", 45)), now) == 1
    assert sc.metrics(now)["n_resolved"] == 1


def test_predictions_are_not_resolved_before_their_target_time(sc):
    sc.log([row("JFK", 20)])
    assert sc.resolve(nas(JFK=("delay", 45)), T0 + 3600) == 0


def test_confusion_matrix_precision_recall_and_baselines(sc):
    # (airport, predicted dep delay, NAS state at the target hour)
    cases = [("JFK", 30, ("delay", 45)), ("EWR", 25, None), ("ATL", 5, ("ground_delay", 90)), ("ORD", 3, None), ("LAX", 40, ("ground_stop", 0)), ("SEA", 2, None)]
    sc.log([row(a, p, event_at_issue=(a in {"JFK", "ATL"})) for a, p, _ in cases])
    now = T0 + 7200 + 120
    sc.resolve(nas(**{a: s for a, _, s in cases if s}), now)
    m = sc.metrics(now)
    assert m["n_resolved"] == 6 and m["confusion"] == {"tp": 2, "fp": 1, "fn": 1, "tn": 2}
    assert m["precision"] == pytest.approx(2 / 3) and m["recall"] == pytest.approx(2 / 3) and m["f1"] == pytest.approx(2 / 3) and m["accuracy"] == pytest.approx(4 / 6)
    assert m["base_rate"] == pytest.approx(3 / 6)
    assert m["baselines"]["always_normal"]["accuracy"] == pytest.approx(3 / 6) and m["baselines"]["always_normal"]["recall"] == 0
    p = m["baselines"]["persistence"]  # predicts an event iff NAS already showed one at issue time: JFK, ATL
    assert p["confusion"] == {"tp": 2, "fp": 0, "fn": 1, "tn": 3} and p["recall"] == pytest.approx(2 / 3)
    lo, hi = m["precision_ci95"]
    assert 0 <= lo < m["precision"] < hi <= 1


def test_minutes_error_is_reported_only_for_real_events_with_a_known_delay(sc):
    sc.log([row("JFK", 30), row("ATL", 10), row("LAX", 40)])
    sc.resolve(nas(JFK=("delay", 50), ATL=("delay", 20), LAX=("ground_stop", 0)), T0 + 7300)
    m = sc.metrics(T0 + 7300)
    assert m["minutes_error"]["n"] == 2 and m["minutes_error"]["mae"] == pytest.approx((20 + 10) / 2)


def test_unresolved_predictions_expire_instead_of_being_guessed(sc):
    sc.log([row("JFK", 30)])
    late = T0 + 7200 + 4 * 3600
    assert sc.resolve(nas(JFK=("delay", 50)), late) == 0
    m = sc.metrics(late)
    assert m["n_expired"] == 1 and m["n_resolved"] == 0 and m["n_pending"] == 0


def test_small_samples_are_flagged_and_text_states_the_ground_truth(sc):
    m = sc.metrics(T0)
    assert m["insufficient_data"] is True and m["n_resolved"] == 0 and m["precision"] is None
    assert "FAA NAS Status" in m["ground_truth"] and "US airports only" in m["ground_truth"]
    assert "15" in m["event_definition"]


def test_wilson_interval_basics():
    lo, hi = wilson(8, 10)
    assert 0.4 < lo < 0.8 < hi < 0.97
    assert wilson(0, 0) is None
    assert wilson(10, 10)[1] == pytest.approx(1.0)


def test_persisted_to_a_file_survives_reopen(tmp_path):
    path = tmp_path / "score.db"
    Scorecard(path).log([row("JFK", 30)])
    assert Scorecard(path).metrics(T0)["n_pending"] == 1
