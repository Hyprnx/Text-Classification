import streamlit as st
from streamlit_extras.switch_page_button import switch_page
st.set_page_config(initial_sidebar_state="collapsed")

st.title("Deep Learning Final Project")
st.subheader(
    """
        The content today are:
    """
)

introduction = st.button("Introduction")
if introduction:
    switch_page("Introduction")

process = st.button("How we did it")
if process:
    switch_page("How we did it")

prod_ready = st.button("The Production-Ready part")
if prod_ready:
    switch_page("The Production-Ready part")

demo = st.button("Demo")
if introduction:
    switch_page("Demo")