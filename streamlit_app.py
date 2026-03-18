"""
Apex Black Box v4.0 – Streamlit entry point.

Uses the Streamlit Component bidirectional protocol (postMessage) to expose
the Python Oracle Engine to the frontend without a separate Flask server.
This approach works on Streamlit Cloud where custom ports are not accessible
from the user's browser.
"""

import shutil
import sys
import tempfile
import traceback
from datetime import datetime, timezone
from pathlib import Path

import streamlit as st
import streamlit.components.v1 as components

from apex_black_box import engine
from apex_black_box.log_utils import (
    ENGINE_VERSION as _ENGINE_VERSION,
    safe_match_id as _safe_match_id,
    append_jsonl as _append_jsonl,
    maybe_log_scan as _maybe_log_scan,
    sanitize_payload as _sanitize_payload,
)

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

# ── Logging helpers ───────────────────────────────────────────────
# All logging functions are imported from apex_black_box.log_utils,
# which is the single source of truth shared with api.py.
# _ENGINE_VERSION, _safe_match_id, _append_jsonl, _maybe_log_scan
# are all available via the imports at the top of this file.


# ── Render component and handle bidirectional requests ───────────
# On each Streamlit run we:
#   1) Render the component, passing any pending scan result back.
#   2) Inspect the component's return value for new scan/final requests.
#   3) If a new (unseen) request arrives, process it and rerun so the
#      result is forwarded to the frontend via pyScanResult.

component_value = _apex_component(
    key="apex_main",
    height=1200,
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
                    _maybe_log_scan(match_id, match_name, _sanitize_payload(payload), result)
                st.session_state["_apex_py_result"] = {
                    "ok": True,
                    "data": result,
                    "reqId": req_id,
                }
            except (ValueError, TypeError, KeyError, RuntimeError) as exc:
                st.session_state["_apex_py_result"] = {
                    "ok": False,
                    "error": str(exc),
                    "reqId": req_id,
                }
            except Exception as exc:
                traceback.print_exc(file=sys.stderr)
                st.session_state["_apex_py_result"] = {
                    "ok": False,
                    "error": f"Internal error: {type(exc).__name__}",
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

