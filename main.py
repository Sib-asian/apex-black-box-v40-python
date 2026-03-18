"""
Apex Black Box v4.0 – legacy entry point kept for backward compatibility.

Preferred command:  streamlit run streamlit_app.py
Equivalent command: streamlit run main.py

All logic lives in streamlit_app.py.  This file imports it so that
existing Streamlit Cloud configurations that point to main.py continue
to work without any changes.

We use ``import streamlit_app`` rather than ``exec`` / ``compile``
because Streamlit's component registry (``declare_component``) inspects
the calling module's ``__name__`` and ``__spec__`` at registration time.
Running the file through ``exec`` leaves those attributes as ``None``,
which raises ``RuntimeError: module is None`` when the component is
first declared.  A normal import preserves all module metadata and
avoids that crash.
"""

import sys
from pathlib import Path

# Ensure the repo root is on sys.path so ``import streamlit_app`` works
# regardless of the working directory Streamlit Cloud uses.
sys.path.insert(0, str(Path(__file__).parent))

import streamlit_app  # noqa: F401, E402
