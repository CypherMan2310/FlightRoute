import numpy as np
import pytest

from aerogrid.ml import conformal as cf


def draw(n, seed, hetero=True):
    rng = np.random.default_rng(seed)
    pred = rng.gamma(2.0, 6.0, n)
    noise = rng.normal(0, 1.0, n) * (1 + (0.5 * pred if hetero else 0))
    return pred, np.maximum(0, pred + noise)


def test_coverage_is_close_to_eighty_percent_on_fresh_data():
    p_cal, y_cal = draw(6000, 0)
    p_new, y_new = draw(20000, 1)
    c = cf.SplitConformal(alpha=0.2).fit(p_cal, y_cal)
    lo, hi = c.interval(p_new)
    cov = cf.coverage(y_new, lo, hi)
    assert 0.77 <= cov <= 0.83


def test_prediction_bins_widen_the_interval_for_larger_predictions():
    p_cal, y_cal = draw(6000, 0)
    c = cf.SplitConformal(alpha=0.2, n_bins=3).fit(p_cal, y_cal)
    lo, hi = c.interval(np.array([3.0, 40.0]))
    assert (hi[1] - lo[1]) > 2 * (hi[0] - lo[0])


def test_bin_conditional_coverage_holds_for_low_and_high_predictions():
    p_cal, y_cal = draw(9000, 0)
    p_new, y_new = draw(30000, 1)
    c = cf.SplitConformal(alpha=0.2, n_bins=3).fit(p_cal, y_cal)
    lo, hi = c.interval(p_new)
    for m in (p_new < np.quantile(p_new, 1 / 3), p_new > np.quantile(p_new, 2 / 3)):
        assert 0.75 <= cf.coverage(y_new[m], lo[m], hi[m]) <= 0.85


def test_interval_is_nonnegative_ordered_and_contains_the_point_prediction_scale():
    p_cal, y_cal = draw(2000, 0)
    c = cf.SplitConformal().fit(p_cal, y_cal)
    p = np.array([0.0, 1.0, 10.0, 50.0])
    lo, hi = c.interval(p)
    assert (lo >= 0).all() and (hi >= lo).all() and (hi >= p).all()


def test_tiny_calibration_sets_fall_back_to_the_widest_residuals():
    c = cf.SplitConformal(alpha=0.2).fit(np.array([5.0, 6.0, 7.0]), np.array([4.0, 9.0, 7.0]))
    lo, hi = c.interval(np.array([6.0]))
    assert lo[0] <= 6.0 - 1.0 + 1e-9 and hi[0] >= 6.0 + 3.0 - 1e-9  # residuals are -1, 0, +3: at least those extremes


def test_save_load_roundtrip(tmp_path):
    p_cal, y_cal = draw(1000, 0)
    c = cf.SplitConformal().fit(p_cal, y_cal)
    c.save(tmp_path / "c.json")
    again = cf.SplitConformal.load(tmp_path / "c.json")
    x = np.array([2.0, 12.0, 30.0])
    assert np.allclose(again.interval(x)[0], c.interval(x)[0]) and np.allclose(again.interval(x)[1], c.interval(x)[1])
    assert again.alpha == 0.2 and again.n_cal == 1000


def test_rejects_bad_alpha_and_empty_calibration():
    with pytest.raises(ValueError):
        cf.SplitConformal(alpha=1.2)
    with pytest.raises(ValueError):
        cf.SplitConformal().fit(np.array([]), np.array([]))
