# main.py

# Importing and initializing the apex_black_box module
import apex_black_box

try:
    from streamlit.runtime.scriptrunner import get_script_run_ctx as _get_ctx
    import streamlit as st
    if _get_ctx() is not None:
        # Running inside a Streamlit server session: render the UI
        ctx = apex_black_box.initialize()
        st.title("Apex Black Box v4.0")
        st.success("initialize() OK")
        st.write(ctx)
except ImportError:
    # streamlit is not installed; Streamlit UI is skipped
    pass

if __name__ == '__main__':
    ctx = apex_black_box.initialize()
    print("initialize() OK:", ctx)