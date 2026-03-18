"""Regression test: importing apex_black_box and apex_black_box.engine must not raise.

Guards against the circular-import that was triggered by
``apex_black_box/__init__.py`` eagerly importing ``apex_black_box.steam``
(which imported ``apex_black_box.engine``) before the package was fully
initialised.

Run with:  pytest tests/test_no_circular_import.py -v
"""

from __future__ import annotations

import importlib
import sys


def _reload_package() -> object:
    """Remove all cached apex_black_box modules and re-import the package.

    This simulates a fresh interpreter import so the test is not a no-op
    when the package has already been imported by a previous test.
    """
    to_remove = [k for k in sys.modules if k == "apex_black_box" or k.startswith("apex_black_box.")]
    for key in to_remove:
        del sys.modules[key]
    return importlib.import_module("apex_black_box")


def test_import_package_does_not_raise():
    """apex_black_box can be imported without errors."""
    _reload_package()


def test_import_engine_after_package_does_not_raise():
    """apex_black_box.engine can be imported after the package without errors."""
    _reload_package()
    importlib.import_module("apex_black_box.engine")


def test_lazy_steam_analyzer_accessible():
    """SteamAnalyzer is accessible from the package via lazy __getattr__."""
    pkg = _reload_package()
    from apex_black_box import SteamAnalyzer  # noqa: F401
    assert SteamAnalyzer is not None
    assert hasattr(pkg, "SteamAnalyzer")


def test_lazy_scan_accessible():
    """scan() is accessible from the package via lazy __getattr__."""
    _reload_package()
    from apex_black_box import scan
    assert callable(scan)


def test_lazy_generate_verdict_accessible():
    """generate_verdict() is accessible from the package via lazy __getattr__."""
    _reload_package()
    from apex_black_box import generate_verdict
    assert callable(generate_verdict)


def test_lazy_enhanced_verdict_accessible():
    """ENHANCED_VERDICT is accessible from the package via lazy __getattr__."""
    pkg = _reload_package()
    # Before access the name must not be cached in the module dict yet.
    assert "ENHANCED_VERDICT" not in pkg.__dict__
    from apex_black_box import ENHANCED_VERDICT
    assert isinstance(ENHANCED_VERDICT, bool)
    # After access it must be cached so subsequent lookups skip __getattr__.
    assert "ENHANCED_VERDICT" in pkg.__dict__


def test_lazy_calibration_dashboard_accessible():
    """CalibrationDashboard is accessible from the package via lazy __getattr__."""
    _reload_package()
    from apex_black_box import CalibrationDashboard
    assert CalibrationDashboard is not None


def test_initialize_returns_all_symbols():
    """initialize() returns a dict containing all public API symbols."""
    pkg = _reload_package()
    result = pkg.initialize()
    assert set(result.keys()) == set(pkg.__all__)
    assert callable(result["scan"])
    assert callable(result["generate_verdict"])
    assert result["SteamAnalyzer"] is not None


def test_unknown_attribute_raises():
    """Accessing an undefined attribute raises AttributeError (not ImportError)."""
    import pytest
    pkg = _reload_package()
    with pytest.raises(AttributeError):
        _ = pkg.nonexistent_symbol_xyz
