"""
Tests for the _normalize_component_msg helper in streamlit_app.py.

This function handles both the direct and wrapped payload forms that
Streamlit component values may take depending on how
``Streamlit.setComponentValue`` is called by the frontend.
"""

from __future__ import annotations

import sys
from pathlib import Path
from unittest.mock import MagicMock

import pytest

# ---------------------------------------------------------------------------
# Import _normalize_component_msg from streamlit_app without triggering
# Streamlit's module-level side-effects (st.set_page_config, component
# registration, shutil.copy2, ...).  We patch the streamlit modules before the
# first import so that all top-level Streamlit calls become no-ops.
# ---------------------------------------------------------------------------

def _import_normalize_fn():
    """Return _normalize_component_msg isolated from Streamlit side-effects."""
    st_mock = MagicMock()
    st_mock.set_page_config = MagicMock()
    st_mock.cache_resource = lambda f: f  # passthrough decorator
    st_mock.session_state = {}

    _orig = {}
    for mod in ("streamlit", "streamlit.components", "streamlit.components.v1"):
        _orig[mod] = sys.modules.get(mod)
        sys.modules[mod] = MagicMock() if mod != "streamlit" else st_mock

    sys.modules.pop("streamlit_app", None)

    repo_root = str(Path(__file__).parent.parent)
    if repo_root not in sys.path:
        sys.path.insert(0, repo_root)

    import streamlit_app  # noqa: PLC0415

    fn = streamlit_app._normalize_component_msg

    for mod, orig in _orig.items():
        if orig is None:
            sys.modules.pop(mod, None)
        else:
            sys.modules[mod] = orig
    sys.modules.pop("streamlit_app", None)

    return fn


_normalize_component_msg = _import_normalize_fn()


class TestNormalizeComponentMsg:
    """Unit tests for _normalize_component_msg."""

    def test_direct_form_passthrough(self):
        """Direct form {action, payload, reqId} must be returned unchanged."""
        msg = {"action": "scan", "payload": {"matchName": "Home v Away"}, "reqId": "r1"}
        assert _normalize_component_msg(msg) is msg

    def test_wrapped_form_unwrapped(self):
        """Wrapped form {value: {action, payload, reqId}} must be unwrapped."""
        inner = {"action": "final", "payload": {"matchName": "Home v Away"}, "reqId": "r2"}
        wrapped = {"value": inner}
        result = _normalize_component_msg(wrapped)
        assert result is inner

    def test_wrapped_non_dict_value_passthrough(self):
        """If 'value' key exists but is not a dict, return the outer dict unchanged."""
        msg = {"value": "some string", "action": "scan", "reqId": "r3"}
        assert _normalize_component_msg(msg) is msg

    def test_empty_dict_passthrough(self):
        """Empty dict must not raise and must be returned as-is."""
        msg: dict = {}
        assert _normalize_component_msg(msg) is msg

    def test_scan_action_preserved(self):
        """Wrapped scan payload must preserve all inner fields after normalisation."""
        inner = {"action": "scan", "payload": {"matchName": "A v B", "tO": 2.5}, "reqId": "abc"}
        result = _normalize_component_msg({"value": inner})
        assert result["action"] == "scan"
        assert result["reqId"] == "abc"
        assert result["payload"]["matchName"] == "A v B"

    def test_final_action_preserved(self):
        """Wrapped final payload must preserve all inner fields after normalisation."""
        inner = {"action": "final", "payload": {"matchName": "X v Y"}, "reqId": "xyz"}
        result = _normalize_component_msg({"value": inner})
        assert result["action"] == "final"
        assert result["reqId"] == "xyz"

    def test_extra_keys_in_wrapper_ignored(self):
        """Extra keys in the outer wrapper are discarded when unwrapping."""
        inner = {"action": "scan", "reqId": "q1", "payload": {}}
        wrapped = {"value": inner, "extra": "noise"}
        result = _normalize_component_msg(wrapped)
        assert result is inner
        assert "extra" not in result
