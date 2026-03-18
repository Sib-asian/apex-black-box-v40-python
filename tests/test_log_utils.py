"""
Unit tests for apex_black_box.log_utils – Supabase-persistence helpers.

Covers:
  - build_match_log_entry: pure function, no side-effects
  - insert_match_log: verifies Supabase table call and graceful failure
"""
from __future__ import annotations

import sys
import threading
import time
import unittest
from unittest.mock import MagicMock, patch

from apex_black_box.log_utils import (
    ENGINE_VERSION,
    build_match_log_entry,
    insert_match_log,
)


class TestBuildMatchLogEntry(unittest.TestCase):
    """build_match_log_entry is a pure helper – no I/O."""

    def _make_payload(self, **kwargs):
        base = {"matchName": "Arsenal vs Chelsea", "hgFT": 2, "agFT": 1}
        base.update(kwargs)
        return base

    def test_required_keys_present(self):
        entry = build_match_log_entry("arsenal_vs_chelsea", "Arsenal vs Chelsea", self._make_payload())
        for key in ("type", "match_id", "matchName", "engine_version", "ts", "hg_ft", "ag_ft", "payload"):
            self.assertIn(key, entry, f"Missing key: {key}")

    def test_type_is_final(self):
        entry = build_match_log_entry("m", "M", self._make_payload())
        self.assertEqual(entry["type"], "final")

    def test_match_id_and_name(self):
        entry = build_match_log_entry("arsenal_vs_chelsea", "Arsenal vs Chelsea", self._make_payload())
        self.assertEqual(entry["match_id"], "arsenal_vs_chelsea")
        self.assertEqual(entry["matchName"], "Arsenal vs Chelsea")

    def test_score_fields(self):
        entry = build_match_log_entry("m", "M", self._make_payload(hgFT=3, agFT=0))
        self.assertEqual(entry["hg_ft"], 3)
        self.assertEqual(entry["ag_ft"], 0)

    def test_score_defaults_to_zero(self):
        entry = build_match_log_entry("m", "M", {})
        self.assertEqual(entry["hg_ft"], 0)
        self.assertEqual(entry["ag_ft"], 0)

    def test_engine_version_default(self):
        entry = build_match_log_entry("m", "M", {})
        self.assertEqual(entry["engine_version"], ENGINE_VERSION)

    def test_engine_version_override(self):
        entry = build_match_log_entry("m", "M", {}, engine_version="vTest")
        self.assertEqual(entry["engine_version"], "vTest")

    def test_ts_is_iso_string(self):
        entry = build_match_log_entry("m", "M", {})
        # Should be parseable as an ISO datetime string ending with UTC offset
        ts = entry["ts"]
        self.assertIsInstance(ts, str)
        self.assertTrue(ts.endswith("+00:00") or ts.endswith("Z"), f"Unexpected ts format: {ts}")

    def test_payload_is_sanitized(self):
        """Only whitelisted fields should appear in the nested payload dict."""
        raw = {"matchName": "X", "hgFT": 1, "agFT": 0, "min": 90, "prevScans": [1, 2, 3]}
        entry = build_match_log_entry("m", "X", raw)
        nested = entry["payload"]
        self.assertIn("min", nested)            # whitelisted field kept
        self.assertNotIn("prevScans", nested)   # non-whitelisted field stripped
        self.assertNotIn("matchName", nested)   # non-whitelisted field stripped

    def test_returns_new_dict_each_call(self):
        """Each call returns an independent dict."""
        e1 = build_match_log_entry("m", "M", {})
        e2 = build_match_log_entry("m", "M", {})
        self.assertIsNot(e1, e2)


class TestInsertMatchLog(unittest.TestCase):
    """insert_match_log delegates to append_jsonl (file + Supabase)."""

    def _sample_entry(self):
        return build_match_log_entry("test_match", "Test Match", {"hgFT": 1, "agFT": 0})

    def test_calls_append_jsonl(self):
        entry = self._sample_entry()
        with patch("apex_black_box.log_utils.append_jsonl") as mock_append:
            insert_match_log("test_match", entry)
            mock_append.assert_called_once_with("test_match", entry)

    def test_does_not_raise_on_append_error(self):
        """insert_match_log must never raise even if append_jsonl fails."""
        entry = self._sample_entry()
        with patch("apex_black_box.log_utils.append_jsonl", side_effect=OSError("disk full")):
            # Should not raise
            insert_match_log("test_match", entry)

    def test_logs_error_to_stderr_on_failure(self):
        entry = self._sample_entry()
        with patch("apex_black_box.log_utils.append_jsonl", side_effect=RuntimeError("boom")):
            with patch("sys.stderr") as mock_stderr:
                insert_match_log("test_match", entry)
                self.assertTrue(mock_stderr.write.called)


class TestSupabaseInsertAsync(unittest.TestCase):
    """_supabase_insert_async fires a background insert when a client exists."""

    def test_supabase_table_insert_called(self):
        """When a Supabase client is available, table('match_logs').insert().execute() is called."""
        from apex_black_box import log_utils

        mock_client = MagicMock()
        with patch.object(log_utils, "_get_supabase", return_value=mock_client):
            log_utils._supabase_insert_async("test_match", {"type": "final"})
            # Allow the daemon thread to run
            time.sleep(0.1)

        mock_client.table.assert_called_once_with("match_logs")
        mock_client.table.return_value.insert.assert_called_once_with(
            {"match_id": "test_match", "entry": {"type": "final"}}
        )
        mock_client.table.return_value.insert.return_value.execute.assert_called_once()

    def test_supabase_insert_error_does_not_propagate(self):
        """An exception inside the background thread must not crash the app."""
        from apex_black_box import log_utils

        mock_client = MagicMock()
        mock_client.table.return_value.insert.return_value.execute.side_effect = Exception("network error")

        with patch.object(log_utils, "_get_supabase", return_value=mock_client):
            log_utils._supabase_insert_async("test_match", {"type": "final"})
            time.sleep(0.1)  # let the thread finish
        # If we reach here without an unhandled exception the test passes.

    def test_no_insert_when_no_client(self):
        """When credentials are absent _supabase_insert_async is a no-op."""
        from apex_black_box import log_utils

        with patch.object(log_utils, "_get_supabase", return_value=None) as mock_get:
            log_utils._supabase_insert_async("test_match", {"type": "final"})
            time.sleep(0.1)
            mock_get.assert_called()


if __name__ == "__main__":
    unittest.main()
