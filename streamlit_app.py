import streamlit as st
import streamlit.components.v1 as components
from pathlib import Path

st.set_page_config(page_title="Apex Black Box v4.0", layout="wide")

html = Path("static/js/V40.html").read_text(encoding="utf-8")
components.html(html, height=980, scrolling=True)
