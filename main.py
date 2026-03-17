import socket
import threading
import streamlit as st
import streamlit.components.v1 as components
from pathlib import Path

st.set_page_config(page_title="Apex Black Box v4.0", layout="wide")


def _find_free_port(start: int = 5050, end: int = 5100) -> int:
    """Return the first free TCP port in [start, end), default 5050."""
    for port in range(start, end):
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
            try:
                s.bind(("127.0.0.1", port))
                return port
            except OSError:
                continue
    return start  # fallback


def _start_flask(port: int) -> None:
    """Start the Flask API server in a daemon thread."""
    try:
        from apex_black_box.api import app
        app.run(host="127.0.0.1", port=port, debug=False, use_reloader=False)
    except Exception as exc:  # pragma: no cover
        print(f"[apex-api] Failed to start Flask backend: {exc}")


# ── Boot Flask backend exactly once per Streamlit session ────────
if "api_port" not in st.session_state:
    port = _find_free_port()
    st.session_state["api_port"] = port
    t = threading.Thread(target=_start_flask, args=(port,), daemon=True)
    t.start()
    print(f"[apex-api] Oracle Engine API listening on http://127.0.0.1:{port}")

api_port = st.session_state["api_port"]

# ── Inject API port into HTML before rendering ───────────────────
html = Path("static/js/V40.html").read_text(encoding="utf-8")
html = html.replace("__APEX_API_PORT__", str(api_port))

components.html(html, height=980, scrolling=True)
