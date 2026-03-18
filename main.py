"""
Apex Black Box v4.0 – legacy entry point kept for backward compatibility.

Preferred command:  streamlit run streamlit_app.py
Equivalent command: streamlit run main.py

All logic lives in streamlit_app.py; this file simply re-executes it so
that existing Streamlit Cloud configurations that point to main.py
continue to work without any changes.
"""

import pathlib
import runpy

runpy.run_path(
    str(pathlib.Path(__file__).parent / "streamlit_app.py"),
    run_name="__main__",
)
