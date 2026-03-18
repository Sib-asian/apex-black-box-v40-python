"""
Apex Black Box v4.0 – Streamlit entry point.

Uses the Streamlit Component bidirectional protocol (postMessage) to expose
the Python Oracle Engine to the frontend without a separate Flask server.
This approach works on Streamlit Cloud where custom ports are not accessible
from the user's browser.
"""

import json
import re
import shutil
import tempfile
import threading
from datetime import datetime, timezone
from pathlib import Path

import streamlit as st
import streamlit.components.v1 as components

from apex_black_box import engine

st.set_page_config(page_title="Apex Black Box v4.0", layout="wide")

# ── Build the Streamlit component once per process ───────────────
# declare_component requires a directory containing index.html.
# We copy static/js/V40.html to a temp dir as index.html so the
# canonical frontend file stays in its original location.


@st.cache_resource
def _create_apex_component():
    """Copy V40.html to a temp dir and declare a bidirectional component."""
    tmp = Path(tempfile.mkdtemp(prefix="apex_component_"))
    src = Path(__file__).parent / "static" / "js" / "V40.html"
    shutil.copy2(src, tmp / "index.html")
    return components.declare_component("apex_bb", path=str(tmp))


_apex_component = _create_apex_component()

# ── Logging helpers (mirrors apex_black_box/api.py) ──────────────
_ENGINE_VERSION = "v40"
_LOG_DIR = Path("data/logs")
_LOG_DIR.mkdir(parents=True, exist_ok=True)

_file_locks: dict[str, threading.Lock] = {}
_file_locks_lock = threading.Lock()


def _get_file_lock(match_id: str) -> threading.Lock:
    with _file_locks_lock:
        if match_id not in _file_locks:
            _file_locks[match_id] = threading.Lock()
        return _file_locks[match_id]


def _safe_match_id(name: str) -> str:
    name = (name or "").strip().lower()
    sanitized = re.sub(r"[^a-z0-9_ -]+", "_", name)
    return re.sub(r"\s+", "_", sanitized)[:120] or "unknown_match"


def _append_jsonl(match_id: str, obj: dict) -> None:
    fp = _LOG_DIR / f"{match_id}.jsonl"
    lock = _get_file_lock(match_id)
    with lock:
        with fp.open("a", encoding="utf-8") as f:
            f.write(json.dumps(obj, ensure_ascii=False) + "\n")


_SNAP_MINUTES = [20, 40, 60, 80]
_SNAP_TAGS = ["snap_20", "snap_40", "snap_60", "snap_80"]

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
    minute = int(payload.get("min", 0))
    hg = int(payload.get("hg", 0))
    ag = int(payload.get("ag", 0))
    rc_h = int(payload.get("rcH", 0))
    rc_a = int(payload.get("rcA", 0))

    tags_to_log: list[str] = []
    state = _get_match_state(match_id)
    with _match_states_lock:
        idx = state["next_snap_idx"]
        while idx < len(_SNAP_MINUTES) and minute >= _SNAP_MINUTES[idx]:
            tags_to_log.append(_SNAP_TAGS[idx])
            idx += 1
        state["next_snap_idx"] = idx

        if state["last_hg"] >= 0 and (hg != state["last_hg"] or ag != state["last_ag"]):
            tags_to_log.append("goal_event")
        if state["last_rcH"] >= 0 and (rc_h != state["last_rcH"] or rc_a != state["last_rcA"]):
            tags_to_log.append("red_event")

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
            "engine_version": _ENGINE_VERSION,
            "ts": ts,
            "minute": minute,
            "payload": payload,
            "engine": engine_data,
        }
        _append_jsonl(match_id, entry)


# ── Render component and handle bidirectional requests ───────────
# On each Streamlit run we:
#   1) Render the component, passing any pending scan result back.
#   2) Inspect the component's return value for new scan/final requests.
#   3) If a new (unseen) request arrives, process it and rerun so the
#      result is forwarded to the frontend via pyScanResult.

component_value = _apex_component(
    key="apex_main",
    height=980,
    pyScanResult=st.session_state.get("_apex_py_result"),
)

if component_value and isinstance(component_value, dict):
    req_id = component_value.get("reqId")
    action = component_value.get("action")
    payload = component_value.get("payload") or {}

    if req_id and req_id != st.session_state.get("_apex_last_req_id"):
        if action == "scan":
            try:
                result = engine.scan(payload)
                match_name = str(payload.get("matchName", ""))
                match_id = _safe_match_id(match_name)
                if match_id and match_id != "unknown_match":
                    _maybe_log_scan(match_id, match_name, payload, result)
                st.session_state["_apex_py_result"] = {
                    "ok": True,
                    "data": result,
                    "reqId": req_id,
                }
            except Exception as exc:
                st.session_state["_apex_py_result"] = {
                    "ok": False,
                    "error": str(exc),
                    "reqId": req_id,
                }
            st.session_state["_apex_last_req_id"] = req_id
            st.rerun()

        elif action == "final":
            match_name = str(payload.get("matchName", ""))
            match_id = _safe_match_id(match_name)
            if match_id and match_id != "unknown_match":
                entry = {
                    "type": "final",
                    "match_id": match_id,
                    "matchName": match_name,
                    "engine_version": _ENGINE_VERSION,
                    "ts": datetime.now(timezone.utc).isoformat(),
                    "hg_ft": int(payload.get("hgFT", 0)),
                    "ag_ft": int(payload.get("agFT", 0)),
                }
                _append_jsonl(match_id, entry)
            st.session_state["_apex_last_req_id"] = req_id

