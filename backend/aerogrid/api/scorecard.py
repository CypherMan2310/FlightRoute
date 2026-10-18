"""The app's own live accuracy: SQLite log of its airport-delay predictions, resolved against FAA NAS Status (US airports only).

Event definition: a model "predicts an event" when its predicted mean departure delay exceeds 15 minutes at the target hour; the
"actual event" is any FAA NAS delay programme (arrival/departure delay, ground delay, ground stop) listed for the airport at that
hour. Predictions are resolved only against live (or very fresh cached) NAS data, never simulated data, and expire if the target
hour was missed. Baselines: always-normal and persistence (NAS already showed an event when the prediction was issued).
"""
from __future__ import annotations

import math
import sqlite3
import threading
from pathlib import Path

EVENT_MIN = 15
EVENT_STATUSES = {"delay", "ground_delay", "ground_stop"}
RESOLVE_WINDOW_S = 1800  # a prediction is resolved within 30 minutes after its target time, else it expires
MAX_CACHED_AGE_S = 300
MIN_RESOLVED = 30
GROUND_TRUTH = "FAA NAS Status airport delay programmes, US airports only (the app's models are not scored outside the US)."
EVENT_DEFINITION = f"predicted mean departure delay > {EVENT_MIN} min vs any FAA delay programme at the target hour"

_SCHEMA = """
CREATE TABLE IF NOT EXISTS predictions (
  id INTEGER PRIMARY KEY AUTOINCREMENT,
  airport TEXT NOT NULL, issued_at REAL NOT NULL, target_at REAL NOT NULL, horizon_h INTEGER NOT NULL,
  pred_dep_min REAL NOT NULL, pred_event INTEGER NOT NULL, nas_event_at_issue INTEGER, trained_on TEXT,
  status TEXT NOT NULL DEFAULT 'pending',
  actual_event INTEGER, actual_status TEXT, actual_delay_min REAL, resolved_at REAL,
  UNIQUE (airport, target_at)
)"""


def wilson(k: int, n: int, z: float = 1.96):
    """Wilson score interval for a proportion; None when n == 0."""
    if n == 0:
        return None
    p = k / n
    denom = 1 + z * z / n
    centre = (p + z * z / (2 * n)) / denom
    half = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / denom
    return max(0.0, centre - half), min(1.0, centre + half)


def _confusion(pairs) -> dict:
    c = {"tp": 0, "fp": 0, "fn": 0, "tn": 0}
    for pred, actual in pairs:
        c[("t" if pred == actual else "f") + ("p" if pred else "n")] += 1
    return c


def _rates(c: dict) -> dict:
    tp, fp, fn, tn = c["tp"], c["fp"], c["fn"], c["tn"]
    n = tp + fp + fn + tn
    prec = tp / (tp + fp) if tp + fp else None
    rec = tp / (tp + fn) if tp + fn else None
    f1 = 2 * tp / (2 * tp + fp + fn) if 2 * tp + fp + fn else None
    return {"n": n, "confusion": c, "precision": prec, "recall": rec, "f1": f1, "accuracy": (tp + tn) / n if n else None,
            "precision_ci95": wilson(tp, tp + fp), "recall_ci95": wilson(tp, tp + fn)}


class Scorecard:
    def __init__(self, path=":memory:"):
        self._lock = threading.Lock()
        self._path = str(path)
        self._conn: sqlite3.Connection | None = None  # opened on first use, so importing the API creates no file

    @property
    def _db(self) -> sqlite3.Connection:
        if self._conn is None:
            if self._path != ":memory:":
                Path(self._path).parent.mkdir(parents=True, exist_ok=True)
            self._conn = sqlite3.connect(self._path, check_same_thread=False)
            self._conn.execute(_SCHEMA)
            self._conn.commit()
        return self._conn

    def log(self, rows: list[dict]) -> int:
        """Insert predictions; a second prediction for the same airport and target hour is ignored. Returns rows inserted."""
        added = 0
        with self._lock:
            for r in rows:
                cur = self._db.execute(
                    "INSERT OR IGNORE INTO predictions (airport, issued_at, target_at, horizon_h, pred_dep_min, pred_event, nas_event_at_issue, trained_on) VALUES (?,?,?,?,?,?,?,?)",
                    (r["airport"], r["issued_at"], r["target_at"], r["horizon_h"], float(r["pred_dep_min"]), int(r["pred_dep_min"] > EVENT_MIN),
                     None if r.get("nas_event_at_issue") is None else int(bool(r["nas_event_at_issue"])), r.get("trained_on")),
                )
                added += cur.rowcount
            self._db.commit()
        return added

    def expire(self, now: float) -> int:
        with self._lock:
            cur = self._db.execute("UPDATE predictions SET status='expired' WHERE status='pending' AND target_at + ? < ?", (RESOLVE_WINDOW_S, now))
            self._db.commit()
            return cur.rowcount

    def resolve(self, nas: dict, now: float) -> int:
        """Resolve predictions whose target time has just passed. Only live (or very fresh cached) NAS data counts."""
        self.expire(now)
        fresh = nas.get("source") == "live" or (nas.get("source") == "cached" and nas.get("age_s", 1e9) <= MAX_CACHED_AGE_S)
        if not fresh:
            return 0
        with self._lock:
            due = self._db.execute("SELECT id, airport FROM predictions WHERE status='pending' AND target_at <= ? AND target_at + ? >= ?", (now, RESOLVE_WINDOW_S, now)).fetchall()
            for pid, airport in due:
                rec = nas.get("airports", {}).get(airport)
                event = bool(rec) and rec.get("status") in EVENT_STATUSES
                self._db.execute(
                    "UPDATE predictions SET status='resolved', actual_event=?, actual_status=?, actual_delay_min=?, resolved_at=? WHERE id=?",
                    (int(event), rec.get("status") if rec else "normal", rec.get("delay_minutes") if rec else 0.0, now, pid),
                )
            self._db.commit()
        return len(due)

    def metrics(self, now: float) -> dict:
        self.expire(now)
        with self._lock:
            res = self._db.execute("SELECT pred_event, actual_event, nas_event_at_issue, pred_dep_min, actual_delay_min FROM predictions WHERE status='resolved'").fetchall()
            counts = dict(self._db.execute("SELECT status, COUNT(*) FROM predictions GROUP BY status").fetchall())
            span = self._db.execute("SELECT MIN(issued_at), MAX(resolved_at) FROM predictions").fetchone()
        model = _rates(_confusion((bool(p), bool(a)) for p, a, *_ in res))
        events = sum(1 for _, a, *_ in res if a)
        n = len(res)
        always = _rates(_confusion((False, bool(a)) for _, a, *_ in res))
        persist = _rates(_confusion((bool(i), bool(a)) for _, a, i, *_ in res if i is not None))
        errs = [abs(pred - actual) for _, a, _, pred, actual in res if a and actual and actual > 0]
        return {
            **{k: model[k] for k in ("precision", "recall", "f1", "accuracy", "confusion", "precision_ci95", "recall_ci95")},
            "n_resolved": n, "n_pending": counts.get("pending", 0), "n_expired": counts.get("expired", 0),
            "base_rate": events / n if n else None, "insufficient_data": n < MIN_RESOLVED,
            "baselines": {"always_normal": {k: always[k] for k in ("accuracy", "recall", "precision", "confusion")}, "persistence": {k: persist[k] for k in ("accuracy", "recall", "precision", "confusion", "n")}},
            "minutes_error": {"n": len(errs), "mae": sum(errs) / len(errs) if errs else None},
            "ground_truth": GROUND_TRUTH, "event_definition": EVENT_DEFINITION, "first_logged": span[0], "last_resolved": span[1],
        }
