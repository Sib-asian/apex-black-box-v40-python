"""
Apex Black Box v4.0 – legacy entry point kept for backward compatibility.

Preferred command:  streamlit run streamlit_app.py
Equivalent command: streamlit run main.py

All logic lives in streamlit_app.py; this file simply re-executes it so
that existing Streamlit Cloud configurations that point to main.py
continue to work without any changes.
"""

import pathlib
import sys

# Ensure the repo root is on the path
sys.path.insert(0, str(pathlib.Path(__file__).parent))

# Execute streamlit_app.py in the correct file context so that
# st.secrets, __file__, and all Streamlit internals work correctly.
_app = pathlib.Path(__file__).parent / "streamlit_app.py"
with open(_app, encoding="utf-8") as _f:
    exec(compile(_f.read(), str(_app), "exec"), {"__file__": str(_app), "__name__": "__main__", "__builtins__": __builtins__})
