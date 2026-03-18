"""Apex Black Box v4.0 – package public API.

Symbols are lazy-loaded on first access to avoid the circular-import that
previously occurred when ``apex_black_box/__init__.py`` eagerly imported
``apex_black_box.steam`` (which in turn imported ``apex_black_box.engine``)
before the package was fully initialised.

All existing import styles continue to work::

    from apex_black_box import SteamAnalyzer   # lazy, via __getattr__
    from apex_black_box import engine           # submodule, always fine
    import apex_black_box; apex_black_box.scan  # lazy, via __getattr__
"""

from __future__ import annotations

import importlib
from typing import Any

__all__ = [
    "SteamAnalyzer",
    "CalibrationDashboard",
    "generate_verdict",
    "ENHANCED_VERDICT",
    "scan",
]

# Maps each exported name to its (relative-module, attribute) source.
_LAZY: dict[str, tuple[str, str]] = {
    "SteamAnalyzer": (".steam", "SteamAnalyzer"),
    "CalibrationDashboard": (".calibration", "CalibrationDashboard"),
    "generate_verdict": (".verdict", "generate_verdict"),
    "ENHANCED_VERDICT": (".verdict", "ENHANCED_VERDICT"),
    "scan": (".engine", "scan"),
}


def __getattr__(name: str):
    """Lazily resolve public API symbols so the package itself stays import-safe."""
    if name in _LAZY:
        module_path, attr = _LAZY[name]
        mod = importlib.import_module(module_path, package=__name__)
        # Cache in the module dict so subsequent accesses bypass __getattr__.
        value = getattr(mod, attr)
        globals()[name] = value
        return value
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


def initialize() -> dict[str, Any]:
    """Import and return all public API symbols. No network calls or side effects."""
    return {name: __getattr__(name) for name in __all__}
