"""Offline training. Not imported at runtime."""
from __future__ import annotations

from dataclasses import dataclass, field

import pandas as pd


@dataclass
class Tables:
    """What train/synth.py and train/real.py both produce; train_all.py consumes only this.

    flights       one row per flown flight (raw weather columns, causes, targets) - see tests/test_synth.py for the schema
    airport_hour  one row per (airport, UTC hour) with the issue-time horizon, raw weather columns and targets y_dep/y_late/y_nas, n
    airports      iata, tier, lat, lon for the graph nodes
    split         epoch-second boundaries: train [start, train_end), calib [calib_start, calib_end), test [test_start, end)
    meta          trained_on ("real" | "synthetic"), window, note, counts
    """

    flights: pd.DataFrame
    airport_hour: pd.DataFrame
    airports: pd.DataFrame
    nodes: list[str]
    split: dict
    meta: dict = field(default_factory=dict)
