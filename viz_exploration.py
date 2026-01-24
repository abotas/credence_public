"""Redirect to the combined app."""
import streamlit as st

st.set_page_config(page_title="Redirect", layout="wide")
st.title("This app has moved")
st.markdown("Please visit: [https://credence.streamlit.app/](https://credence.streamlit.app/)")
st.link_button("Go to new app", "https://credence.streamlit.app/")
