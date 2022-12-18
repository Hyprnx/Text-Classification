import streamlit as st
from system_profile_utilities import get_system_info
import json

st.set_page_config(initial_sidebar_state="collapsed",
                   page_title="Text Classification",
                   page_icon="📑",
                   )

st.header("Other pages")
st.subheader(""" This page contains the other information of the project. """)
st.json(json.loads(get_system_info()))