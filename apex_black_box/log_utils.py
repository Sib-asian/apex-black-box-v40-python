"""
Shared logging helpers used by both api.py (Flask) and streamlit_app.py.

Single source of truth for JSONL logging, match-state tracking and snap
scheduling.  Neither api.py nor streamlit_app.py should redefine these
functions — import them from here instead.

No circular imports: this module must NOT import from engine.py or verdict.py.
"""
from __future__ import annotations

import json
import re
import threading
from datetime import datetime, timezone
from pathlib import Path

ENGINE_VERSION = "v40"
LOG_DIR = Path("data/logs")
LOG_DIR.mkdir(parents=True, exist_ok=True)

# Snapshot minutes — kept in sync between Flask API and Streamlit.
# This module is the single source of truth — do not redefine in api.py or streamlit_app.py.
SNAP_MINUTES = [5, 10, 20, 40, 60, 80, 88]
SNAP_TAGS = ["snap_5", "snap_10", "snap_20", "snap_40", "snap_60", "snap_80", "snap_88"]

# One lock per file path (keyed by match_id) to serialize JSONL appends.
_file_locks: dict[str, threading.Lock] = {}
_file_locks_lock = threading.Lock()

# Per-match scan state for snapshot + event dedup.
_match_states: dict[str, dict] = {}
_match_states_lock = threading.Lock()


def get_file_lock(match_id: str) -> threading.Lock:
    """Return (creating if needed) a per-file threading.Lock for match_id."""
    with _file_locks_lock:
        if match_id not in _file_locks:
            _file_locks[match_id] = threading.Lock()
        return _file_locks[match_id]


def safe_match_id(name: str) -> str:
    """Sanitize a match name to a safe file-system identifier."""
    name = (name or "").strip().lower()
    sanitized = re.sub(r"[^a-z0-9_ -]+", "_", name)
    return re.sub(r"\s+", "_", sanitized)[:120] or "unknown_match"


def append_jsonl(match_id: str, obj: dict) -> None:
    """Thread-safely append a JSON line to data/logs/<match_id>.jsonl."""
    fp = LOG_DIR / f"{match_id}.jsonl"
    lock = get_file_lock(match_id)
    with lock:
        with fp.open("a", encoding="utf-8") as f:
            f.write(json.dumps(obj, ensure_ascii=False) + "\n")


def get_match_state(match_id: str) -> dict:
    """Return (creating if needed) the mutable state dict for match_id."""
    with _match_states_lock:
        if match_id not in _match_states:
            _match_states[match_id] = {
                "next_snap_idx": 0,
                "last_hg": -1,
                "last_ag": -1,
                "last_rcH": -1,
                "last_rcA": -1,
            }
        return _match_states[match_id]


def maybe_log_scan(
    match_id: str,
    match_name: str,
    payload: dict,
    result: dict,
    engine_version: str = ENGINE_VERSION,
) -> None:
    """Decide whether the current scan should be logged and write JSONL if so.

    Logs on snapshot thresholds (SNAP_MINUTES), goal events, and red-card
    events.  Callers must sanitize the payload before passing it in if they
    want to avoid logging sensitive or large fields (e.g. prevScans).
    """
    minute = int(payload.get("min", 0))
    hg = int(payload.get("hg", 0))
    ag = int(payload.get("ag", 0))
    rc_h = int(payload.get("rcH", 0))
    rc_a = int(payload.get("rcA", 0))

    tags_to_log: list[str] = []

    state = get_match_state(match_id)
    with _match_states_lock:
        # Snapshot thresholds: log first scan that reaches each threshold minute
        idx = state["next_snap_idx"]
        while idx < len(SNAP_MINUTES) and minute >= SNAP_MINUTES[idx]:
            tags_to_log.append(SNAP_TAGS[idx])
            idx += 1
        state["next_snap_idx"] = idx

        # Goal event: score changed since last logged state
        if state["last_hg"] >= 0 and (hg != state["last_hg"] or ag != state["last_ag"]):
            tags_to_log.append("goal_event")

        # Red card event
        if state["last_rcH"] >= 0 and (rc_h != state["last_rcH"] or rc_a != state["last_rcA"]):
            tags_to_log.append("red_event")

        # Update last known state
        state["last_hg"] = hg
        state["last_ag"] = ag
        state["last_rcH"] = rc_h
        state["last_rcA"] = rc_a

    if not tags_to_log:
        return

    ts = datetime.now(timezone.utc).isoformat()
    engine_data = {
        "probs": result.get("probs"),
        "confidence": result.get("confidence"),
        "dynamics": result.get("dynamics"),
        "metrics": result.get("metrics"),
        "raw": result.get("raw"),
    }

    for tag in tags_to_log:
        entry = {
            "type": "scan",
            "tag": tag,
            "match_id": match_id,
            "matchName": match_name,
            "engine_version": engine_version,
            "ts": ts,
            "minute": minute,
            "payload": payload,
            "engine": engine_data,
        }
        append_jsonl(match_id, entry)
