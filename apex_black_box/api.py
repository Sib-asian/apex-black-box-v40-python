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

import time
import threading
from datetime import datetime, timezone

from flask import Flask, request, jsonify
from flask_cors import CORS

from apex_black_box import engine
from apex_black_box.log_utils import (
    ENGINE_VERSION,
    safe_match_id as _safe_match_id,
    append_jsonl as _append_jsonl,
    maybe_log_scan as _log_utils_maybe_log_scan,
)

app = Flask(__name__)
CORS(app)  # allow same-machine requests from the Streamlit iframe

# ── Logging setup ────────────────────────────────────────────────
# Logging helpers (_safe_match_id, _append_jsonl, SNAP_MINUTES, SNAP_TAGS, etc.)
# are imported from apex_black_box.log_utils — single source of truth shared with
# streamlit_app.py.


# ── FIX 11: Payload sanitization ─────────────────────────────────
# Whitelist of fields to include in the logged payload.
_PAYLOAD_WHITELIST = frozenset([
    "min", "rec", "hg", "ag", "lastGoal",
    "sotH", "misH", "corH", "daH",
    "sotA", "misA", "corA", "daA",
    "rcH", "rcA", "tC", "tO", "sC", "sO",
    "isKnockout", "possH", "possA",
])
_MAX_STRING_LENGTH = 200  # max characters for any string field in sanitized payload


def _sanitize_payload(payload: dict) -> dict:
    """Return a sanitized copy of payload with only whitelisted fields.

    Truncates string values to _MAX_STRING_LENGTH chars and omits large fields like prevScans.
    """
    result: dict = {}
    for key in _PAYLOAD_WHITELIST:
        if key in payload:
            val = payload[key]
            if isinstance(val, str):
                val = val[:_MAX_STRING_LENGTH]
            result[key] = val
    return result


# ── FIX 9: Per-match rate limiting ───────────────────────────────
_RATE_LIMIT_WINDOW: float = 1.0   # seconds
_RATE_LIMIT_MAX: int = 10         # max requests per window per match_id
_rate_buckets: dict[str, list[float]] = {}
_rate_lock = threading.Lock()


def _check_rate_limit(match_id: str) -> bool:
    """Returns True if request is allowed, False if rate limited."""
    now = time.monotonic()
    with _rate_lock:
        bucket = _rate_buckets.setdefault(match_id, [])
        # Remove timestamps older than the window
        _rate_buckets[match_id] = [t for t in bucket if now - t < _RATE_LIMIT_WINDOW]
        if len(_rate_buckets[match_id]) >= _RATE_LIMIT_MAX:
            return False
        _rate_buckets[match_id].append(now)
        return True


def _maybe_log_scan(match_id: str, match_name: str, payload: dict, result: dict) -> None:
    """Delegate to log_utils.maybe_log_scan with sanitized payload."""
    # FIX 11: log only the sanitized subset of the payload (no prevScans, no steam quotes)
    sanitized = _sanitize_payload(payload)
    _log_utils_maybe_log_scan(match_id, match_name, sanitized, result, ENGINE_VERSION)


# ── Routes ───────────────────────────────────────────────────────

@app.route("/api/health", methods=["GET"])
def health():
    return jsonify({"status": "ok"})


@app.route("/api/scan", methods=["POST"])
def scan():
    payload = request.get_json(force=True, silent=True) or {}
    match_name = str(payload.get("matchName", ""))
    match_id = _safe_match_id(match_name)

    # FIX 9: Rate limit per match_id to prevent runaway frontend loops
    if match_id and not _check_rate_limit(match_id):
        return jsonify({"error": "Too many requests"}), 429

    try:
        result = engine.scan(payload)
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
