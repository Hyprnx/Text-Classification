import streamlit as st
from streamlit_extras.switch_page_button import switch_page

st.header("The \" Production Ready\" process")

next_page = st.button("Demo")
if next_page:
    switch_page("Demo")
