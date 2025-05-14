# app/streamlit_app.py

from app.ui import render_app

render_app()

if __name__ == "__main__":
    import streamlit as st
    st.warning("⚠️ Please run this app via: `streamlit run run_app.py`")
