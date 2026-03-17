"""
Flask API – exposes the Oracle Engine as a local HTTP endpoint.

POST /api/scan
  Body: JSON payload (see apex_black_box/engine.py for field docs)
  Returns: JSON result from engine.scan()

POST /api/final
  Body: { matchName: str, hgFT: int, agFT: int }
  Returns: { status: "ok" }

GET /api/health
  Returns: {"status": "ok"}
"""

import json
import re
import threading
from datetime import datetime, timezone
from pathlib import Path

from flask import Flask, request, jsonify
from flask_cors import CORS

from apex_black_box import engine

app = Flask(__name__)
CORS(app)  # allow same-machine requests from the Streamlit iframe

# ── Logging setup ────────────────────────────────────────────────
ENGINE_VERSION = "v40"
LOG_DIR = Path("data/logs")
LOG_DIR.mkdir(parents=True, exist_ok=True)

# One lock per file path (keyed by match_id) to serialize JSONL appends.
_file_locks: dict[str, threading.Lock] = {}
_file_locks_lock = threading.Lock()


def _get_file_lock(match_id: str) -> threading.Lock:
    with _file_locks_lock:
        if match_id not in _file_locks:
            _file_locks[match_id] = threading.Lock()
        return _file_locks[match_id]


def _safe_match_id(name: str) -> str:
    """Sanitize a match name to a safe file-system identifier."""
    name = (name or "").strip().lower()
    sanitized = re.sub(r"[^a-z0-9_ -]+", "_", name)
    return re.sub(r"\s+", "_", sanitized)[:120] or "unknown_match"


def _append_jsonl(match_id: str, obj: dict) -> None:
    """Thread-safely append a JSON line to data/logs/<match_id>.jsonl."""
    fp = LOG_DIR / f"{match_id}.jsonl"
    lock = _get_file_lock(match_id)
    with lock:
        with fp.open("a", encoding="utf-8") as f:
            f.write(json.dumps(obj, ensure_ascii=False) + "\n")


# ── Per-match scan-state for schema-4 snapshot + event dedup ─────
_SNAP_MINUTES = [20, 40, 60, 80]
_SNAP_TAGS = ["snap_20", "snap_40", "snap_60", "snap_80"]

# match_id -> { next_snap_idx, last_hg, last_ag, last_rcH, last_rcA }
_match_states: dict[str, dict] = {}
_match_states_lock = threading.Lock()


def _get_match_state(match_id: str) -> dict:
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


def _maybe_log_scan(match_id: str, match_name: str, payload: dict, result: dict) -> None:
    """Decide whether the current scan should be logged and write JSONL if so."""
    minute = int(payload.get("min", 0))
    hg = int(payload.get("hg", 0))
    ag = int(payload.get("ag", 0))
    rc_h = int(payload.get("rcH", 0))
    rc_a = int(payload.get("rcA", 0))

    tags_to_log: list[str] = []

    state = _get_match_state(match_id)
    with _match_states_lock:
        # Snapshot thresholds: log first scan that reaches each threshold minute
        idx = state["next_snap_idx"]
        while idx < len(_SNAP_MINUTES) and minute >= _SNAP_MINUTES[idx]:
            tags_to_log.append(_SNAP_TAGS[idx])
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
            "engine_version": ENGINE_VERSION,
            "ts": ts,
            "minute": minute,
            "payload": payload,
            "engine": engine_data,
        }
        _append_jsonl(match_id, entry)


# ── Routes ───────────────────────────────────────────────────────

@app.route("/api/health", methods=["GET"])
def health():
    return jsonify({"status": "ok"})


@app.route("/api/scan", methods=["POST"])
def scan():
    payload = request.get_json(force=True, silent=True) or {}
    try:
        result = engine.scan(payload)
        match_name = str(payload.get("matchName", ""))
        match_id = _safe_match_id(match_name)
        if match_id and match_id != "unknown_match":
            _maybe_log_scan(match_id, match_name, payload, result)
        return jsonify(result)
    except Exception as exc:  # pragma: no cover
        return jsonify({"error": str(exc)}), 500


@app.route("/api/final", methods=["POST"])
def final():
    payload = request.get_json(force=True, silent=True) or {}
    match_name = str(payload.get("matchName", ""))
    match_id = _safe_match_id(match_name)

    try:
        hg_ft = int(payload["hgFT"])
        ag_ft = int(payload["agFT"])
    except (KeyError, ValueError, TypeError):
        return jsonify({"error": "Invalid or missing hgFT/agFT"}), 400

    entry = {
        "type": "final",
        "match_id": match_id,
        "matchName": match_name,
        "engine_version": ENGINE_VERSION,
        "ts": datetime.now(timezone.utc).isoformat(),
        "hg_ft": hg_ft,
        "ag_ft": ag_ft,
    }
    _append_jsonl(match_id, entry)
    return jsonify({"status": "ok"})
