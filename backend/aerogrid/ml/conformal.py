"""Split-conformal prediction intervals (80% by default) on signed residuals, stratified by predicted-delay bin (Mondrian).

Delay is skewed and its spread grows with the prediction, so one global half-width would be wrong at both ends. Each bin of
the calibration set gets its own lower/upper residual quantiles with the finite-sample correction; bins with fewer than
`min_bin` calibration rows borrow the pooled quantiles. Calibration data must never be used for fitting or model selection.
"""
from __future__ import annotations

import json
import math
from pathlib import Path

import numpy as np


def coverage(y, lo, hi) -> float:
    y = np.asarray(y)
    return float(np.mean((y >= np.asarray(lo)) & (y <= np.asarray(hi))))


def _quantiles(resid: np.ndarray, alpha: float) -> tuple[float, float]:
    r = np.sort(resid)
    n = len(r)
    k_hi = min(n, math.ceil((n + 1) * (1 - alpha / 2)))  # k > n would mean +inf: use the largest residual
    k_lo = max(1, math.floor((n + 1) * alpha / 2))  # k = 0 would mean -inf: use the smallest residual
    return float(r[k_lo - 1]), float(r[k_hi - 1])


class SplitConformal:
    def __init__(self, alpha: float = 0.2, n_bins: int = 3, min_bin: int = 50):
        if not 0 < alpha < 1:
            raise ValueError("alpha must be in (0, 1)")
        self.alpha, self.n_bins, self.min_bin = alpha, n_bins, min_bin
        self.edges: list[float] = []
        self.bins: list[tuple[float, float]] = []
        self.pooled: tuple[float, float] = (0.0, 0.0)
        self.n_cal = 0

    def fit(self, pred, y) -> "SplitConformal":
        pred, y = np.asarray(pred, dtype=float), np.asarray(y, dtype=float)
        if len(pred) == 0:
            raise ValueError("empty calibration set")
        resid = y - pred
        self.n_cal = len(pred)
        self.pooled = _quantiles(resid, self.alpha)
        self.edges = [float(e) for e in np.quantile(pred, np.linspace(0, 1, self.n_bins + 1)[1:-1])] if self.n_bins > 1 else []
        which = np.searchsorted(self.edges, pred, side="right")
        self.bins = [_quantiles(resid[which == b], self.alpha) if (which == b).sum() >= self.min_bin else self.pooled for b in range(self.n_bins)]
        return self

    def interval(self, pred) -> tuple[np.ndarray, np.ndarray]:
        pred = np.asarray(pred, dtype=float)
        q = np.array(self.bins)[np.searchsorted(self.edges, pred, side="right")]
        lo = np.maximum(0.0, pred + q[:, 0])
        hi = np.maximum.reduce([pred + q[:, 1], lo, pred])
        return lo, hi

    def save(self, path) -> None:
        Path(path).write_text(json.dumps({"alpha": self.alpha, "n_bins": self.n_bins, "min_bin": self.min_bin, "edges": self.edges, "bins": self.bins, "pooled": self.pooled, "n_cal": self.n_cal}), encoding="utf-8")

    @classmethod
    def load(cls, path) -> "SplitConformal":
        d = json.loads(Path(path).read_text(encoding="utf-8"))
        c = cls(d["alpha"], d["n_bins"], d["min_bin"])
        c.edges, c.bins, c.pooled, c.n_cal = d["edges"], [tuple(b) for b in d["bins"]], tuple(d["pooled"]), d["n_cal"]
        return c
